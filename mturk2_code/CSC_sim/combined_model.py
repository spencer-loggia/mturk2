import torch
from torch import nn

from CogSSM import SSM, SSMConfig, InferenceCache
from dataclasses import dataclass
from typing import List, Tuple
from einops import rearrange


@dataclass
class AgentConfig(SSMConfig):
    percept_dim: int = 2
    reward_dim: int = 1
    action_dims: Tuple[int] = (1,)  #
    device: str = "cpu"

    def __post_init__(self):
        super().__post_init__()
        self.obs_dim = self.percept_dim + self.reward_dim
        if len(self.action_dims) < 1:
            raise ValueError("need to have at least 1 output set per node")


import torch
import torch.nn as nn

class Deform(nn.Module):
    """
    Smooth deformation y = x + W2 * tanh(W1 x).
    With `groups`, applies independent deformations to subsets of channels.

    Examples:
      - groups=1        -> single transform over all channels (original behavior)
      - groups=channels -> separate transform per channel
      - channels=4, groups=2 -> one transform over ch[0:2], another over ch[2:4]
    """

    def __init__(
        self,
        channels: int = 4,
        deform_basis: int = 8,
        box = (-1.0, 1.0),
        reg_weight: float = 1.0,
        sobol_samples: int = 1024,
        sobol_scramble: bool = True,
        margin_sigma_min: float = 1e-2,   # penalize if smallest singular value < margin
        groups: int = 1,
        device="cpu"
    ):
        super().__init__()
        self.activ = nn.Tanh()
        self.channels = int(channels)
        self.deform_basis = int(deform_basis)
        self.reg_weight = float(reg_weight)
        self.sobol_samples = int(sobol_samples)
        self.sobol_scramble = bool(sobol_scramble)
        self.margin_sigma_min = float(margin_sigma_min)

        # --- Groups setup ---
        self.groups = int(groups)
        if self.groups < 1:
            raise ValueError("groups must be >= 1")
        if self.channels % self.groups != 0:
            raise ValueError(f"channels ({self.channels}) must be divisible by groups ({self.groups})")
        self.group_size = self.channels // self.groups
        self.device = device

        # Register the domain as buffers so dtype/device track the module
        box = torch.as_tensor(box, dtype=torch.float32, device=self.device)
        domain = torch.stack([box for _ in range(self.channels)], dim=0)  # (C, 2)
        self.register_buffer("domain", domain, persistent=False)  # (channels, 2)

        # SobolEngine will be created lazily because it needs correct device/dtype
        self._sobol = None
        self._sobol_dimension = None

        # Parameters: either a single pair of Linear layers, or one pair per group
        if self.groups == 1:
            self.project  = nn.Linear(self.channels, self.deform_basis, bias=False, device=device)  # W1: (B, C)
            self.collapse = nn.Linear(self.deform_basis, self.channels, bias=False, device=device)  # W2: (C, B)
            # Small, stable init keeps Jacobian near I initially
            nn.init.kaiming_uniform_(self.project.weight, a=5**0.5)
            nn.init.kaiming_uniform_(self.collapse.weight, a=5**0.5)
            with torch.no_grad():
                self.project.weight.mul_(0.2)
                self.collapse.weight.mul_(0.2)
        else:
            self.project = nn.ModuleList([
                nn.Linear(self.group_size, self.deform_basis, bias=False, device=device)
                for _ in range(self.groups)
            ])
            self.collapse = nn.ModuleList([
                nn.Linear(self.deform_basis, self.group_size, bias=False, device=device)
                for _ in range(self.groups)
            ])
            for p, c in zip(self.project, self.collapse):
                nn.init.kaiming_uniform_(p.weight, a=5**0.5)
                nn.init.kaiming_uniform_(c.weight, a=5**0.5)
                with torch.no_grad():
                    p.weight.mul_(0.2)
                    c.weight.mul_(0.2)

    def _sobol_engine(self, device, dtype):
        if self._sobol is None or self._sobol_dimension != self.channels:
            # Create per-device engine; store dimension to detect changes
            self._sobol_dimension = self.channels
            self._sobol = torch.quasirandom.SobolEngine(
                dimension=self.channels, scramble=self.sobol_scramble,
            )
        # SobolEngine only returns float32 CPU; we’ll cast after draw.
        return self._sobol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels)
        """
        if self.groups == 1:
            h = self.activ(self.project(x))
            return x + self.collapse(h)

        # Grouped path: split channels, transform independently, then concat
        xs = x.split(self.group_size, dim=-1)
        ys = []
        for g, (proj, coll) in enumerate(zip(self.project, self.collapse)):
            h = self.activ(proj(xs[g]))           # (batch, B)
            yg = xs[g] + coll(h)                  # (batch, group_size)
            ys.append(yg)
        return torch.cat(ys, dim=-1)

    @torch.no_grad()
    def _sample_box(self, n: int, device, dtype) -> torch.Tensor:
        # Draw Sobol samples in [0,1]^C then affine map to the domain per-dim
        se = self._sobol_engine(device, dtype)
        u01 = se.draw(n)  # (n, C), float32 on CPU
        u01 = u01.to(device=self.device, dtype=dtype)
        lo, hi = self.domain[:, 0], self.domain[:, 1]          # (C,), (C,)
        return u01 * (hi - lo).unsqueeze(0) + lo.unsqueeze(0)  # (n, C)

    def compute_penalty(self) -> torch.Tensor:
        """
        Jacobian regularizer:
          For each Sobol sample, compute the (approx) Jacobian J = I + W2 * diag(sech^2(W1 x)) * W1,
          penalize when the smallest singular value of J falls below margin_sigma_min.
        With groups > 1, J is block-diagonal; we aggregate singular values across blocks.
        """
        if self.reg_weight <= 0:
            # Create a scalar 0 with correct dtype/device
            if self.groups == 1:
                return self.project.weight.new_zeros(())
            else:
                return self.project[0].weight.new_zeros(())


        # Device/dtype from params
        device = self.device
        if self.groups == 1:
            dtype = self.project.weight.dtype
        else:
            dtype = self.project[0].weight.dtype

        C = self.channels
        G = self.groups
        gs = self.group_size

        x = self._sample_box(self.sobol_samples, device, dtype)  # (N, C)

        if G == 1:
            W1 = self.project.weight  # (B, C)
            W2 = self.collapse.weight  # (C, B)

            z = x @ W1.t()                  # (N, B)
            t = torch.tanh(z)               # (N, B)
            sech2 = 1.0 - t * t             # (N, B)

            # J = I + W2 * diag(sech2) * W1, arranged as batch matmul:
            W2_scaled = W2.unsqueeze(0) * sech2.unsqueeze(1)  # (N, C, B)
            J = W2_scaled @ W1.unsqueeze(0)                   # (N, C, C)
            I = torch.eye(C, device=device, dtype=dtype).expand_as(J)
            J = I + J

            svals = torch.linalg.svdvals(J)  # (N, C)
        else:
            # Compute per-group Jacobian blocks and collect singular values
            x_groups = x.split(gs, dim=-1)  # list of (N, gs)
            svals_list = []
            eye_cache = None
            for g in range(G):
                W1g = self.project[g].weight        # (B, gs)
                W2g = self.collapse[g].weight       # (gs, B)

                zg = x_groups[g] @ W1g.t()          # (N, B)
                tg = torch.tanh(zg)                 # (N, B)
                sech2g = 1.0 - tg * tg              # (N, B)

                W2s = W2g.unsqueeze(0) * sech2g.unsqueeze(1)  # (N, gs, B)
                Jg = W2s @ W1g.unsqueeze(0)                   # (N, gs, gs)

                if eye_cache is None or eye_cache.size(-1) != gs:
                    eye_cache = torch.eye(gs, device=device, dtype=dtype)
                Jg = Jg + eye_cache.expand_as(Jg)

                svals_g = torch.linalg.svdvals(Jg)  # (N, gs)
                svals_list.append(svals_g)

            svals = torch.cat(svals_list, dim=-1)   # (N, C) — concat of block singular values

        smin = svals[..., -1]  # (N,) smallest singular value per sample
        cost = torch.nn.functional.softplus(self.margin_sigma_min - smin).mean()
        return self.reg_weight * cost

    def to(self, *args, **kwargs):
        """
        Move module params/buffers and update self.device.
        Also resets the Sobol engine so it can be lazily recreated on the new device.
        """
        ret = super().to(*args, **kwargs)

        # Parse torch's flexible .to(...) signature
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device is None and len(args) >= 1 and not isinstance(args[0], torch.dtype):
            device = args[0]
        if dtype is None and len(args) >= 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]

        if device is not None:
            self.device = torch.device(device)

        # SobolEngine lives on CPU and we cast samples afterward; safest to recreate after a move
        self._sobol = None
        self._sobol_dimension = None

        return ret

# ---------------------------------------------------------------------------
# Model: Q‑value network with internal SSM state + temperature scaling
# ---------------------------------------------------------------------------

class QAgent(torch.nn.Module):
    """Maps (sin,cos,reward) features -> scalar Q‑value for each option.

    This is the cognitive head of the model

    The policy is derived via soft‑max(Q / τ) with learnable temperature τ.
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.obs_dim = config.obs_dim
        self.config = config
        self.ssm = SSM(config, device=config.device)
        initial_vals = torch.empty((config.nheads, config.headdim,  config.d_state),
                                   dtype=torch.float32, device=config.device)
        self.initial_states = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(initial_vals * .1))
        # self.sa = torch.tensor([0.], device=config.device)
        # self.sa_net = torch.nn.Linear(in_features=self.ssm.args.d_inner * self.ssm.args.d_state, out_features=config)
        self.hidden = None  # b, p, h, s

        # Inference cache for sequential rollout
        self.sequential = False
        self.cache = None

    # ---------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, reward: torch.Tensor, *, k: int):
        """obs: (T, B*k, obs_dim), reward: (T, B*k, 1)   → values (T,B,k) and logits (T,B,k)"""
        T, Bk, _ = obs.shape

        # if operating with self action need to add action to network.
        if self.cache is None:
            # start from the top
            cache = InferenceCache.alloc(batch_size=Bk, args=self.config, device=self.config.device)
            cache.ssm_state = cache.ssm_state + self.initial_states.unsqueeze(0)
            self.cache = cache

        x = obs.float().view(T * Bk, -1)  # (T*Bk,input_dim)
        rew = reward.float().view(T * Bk, 1)  # (T*Bk,input_dim)
        x = rearrange(x, '(t b) s -> t b s', t=T, b=Bk)  # (T,Bk,N_UNITS)
        rew = rearrange(rew, '(t b) 1 -> t b 1', b=Bk)

        y, self.cache = self.ssm(x.transpose(0, 1), rew.transpose(0, 1), self.cache)  # (T,Bk,N_UNITS)
        self.hidden = self.cache.ssm_state.clone()
        y = y.transpose(0, 1)

        q = y  # (T,Bk)
        q = q.view(T, -1, k)  # (T,B,k)

        logits = q
        return q, logits

    # ---------------------------------------------------------------------
    def reset(self):
        self.sequential = False
        self.cache = None

    def to(self, *args, **kwargs):
        """
        Move the agent, its SSM, and (if present) the inference cache.
        Also keeps AgentConfig.device in sync.
        """
        ret = super().to(*args, **kwargs)

        # Parse torch's flexible .to(...) signature
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device is None and len(args) >= 1 and not isinstance(args[0], torch.dtype):
            device = args[0]
        if dtype is None and len(args) >= 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]

        if device is not None:
            # keep config + internal flags consistent
            dev = torch.device(device)
            self.config.device = str(dev)
            self.ssm = self.ssm.to(device)
            # if there's a live cache, move it too
            if self.cache is not None:
                # InferenceCache in your project already has .to(...) from earlier
                # If not, you can manually move tensors:
                #   self.cache.conv_state = self.cache.conv_state.to(dev, dtype=dtype)
                #   self.cache.ssm_state  = self.cache.ssm_state.to(dev, dtype=dtype)
                self.cache = self.cache.to(dev, dtype=dtype)

        return ret
