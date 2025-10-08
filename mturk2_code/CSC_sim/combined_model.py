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


class Deform(nn.Module):
    """
    Smooth deformation y = x + W2 * tanh(W1 x).
    Intended to model a low-amplitude, smooth perceptual bias while remaining close
    to a diffeomorphism on a bounded box via a Jacobian regularizer.
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
    ):
        super().__init__()
        self.activ = nn.Tanh()
        self.channels = int(channels)
        self.project  = nn.Linear(channels, deform_basis, bias=False)  # W1: (B, C)
        self.collapse = nn.Linear(deform_basis, channels, bias=False)  # W2: (C, B)
        self.reg_weight = float(reg_weight)
        self.sobol_samples = int(sobol_samples)
        self.sobol_scramble = bool(sobol_scramble)
        self.margin_sigma_min = float(margin_sigma_min)

        # Register the domain as buffers so dtype/device track the module
        box = torch.as_tensor(box, dtype=torch.float32)
        domain = torch.stack([box for _ in range(channels)], dim=0)  # (C, 2)
        self.register_buffer("domain", domain, persistent=False)  # (channels, 2)

        # SobolEngine will be created lazily because it needs correct device/dtype
        self._sobol = None

        # Small, stable init keeps Jacobian near I initially
        nn.init.kaiming_uniform_(self.project.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.collapse.weight, a=5**0.5)
        with torch.no_grad():
            self.project.weight.mul_(0.2)
            self.collapse.weight.mul_(0.2)

    def _sobol_engine(self, device, dtype):
        if self._sobol is None or self._sobol_dimension != self.channels:
            # Create per-device engine; store dimension to detect changes
            self._sobol_dimension = self.channels
            self._sobol = torch.quasirandom.SobolEngine(
                dimension=self.channels, scramble=self.sobol_scramble
            )
        # SobolEngine only returns float32 CPU; we’ll cast after draw.
        return self._sobol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels)
        """
        h = self.activ(self.project(x))
        return x + self.collapse(h)

    @torch.no_grad()
    def _sample_box(self, n: int, device, dtype) -> torch.Tensor:
        # Draw Sobol samples in [0,1]^C then affine map to the domain per-dim
        se = self._sobol_engine(device, dtype)
        u01 = se.draw(n)  # (n, C), float32 on CPU
        u01 = u01.to(device=device, dtype=dtype)
        lo, hi = self.domain[:, 0], self.domain[:, 1]          # (C,), (C,)
        return u01 * (hi - lo).unsqueeze(0) + lo.unsqueeze(0)  # (n, C)

    def compute_penalty(self) -> torch.Tensor:
        if self.reg_weight <= 0:
            return self.project.weight.new_zeros(())

        device = self.project.weight.device
        dtype = self.project.weight.dtype
        C = self.channels

        x = self._sample_box(self.sobol_samples, device, dtype)  # (N, C)

        W1 = self.project.weight  # (B, C)
        W2 = self.collapse.weight  # (C, B)
        z = x @ W1.t()  # (N, B)
        t = torch.tanh(z)  # (N, B)
        sech2 = 1.0 - t * t  # (N, B), stable sech^2

        W2_scaled = W2.unsqueeze(0) * sech2.unsqueeze(1)  # (N, C, B)

        J = W2_scaled @ W1.unsqueeze(0)  # (N, C, C)
        I = torch.eye(C, device=device, dtype=dtype).expand_as(J)
        J = I + J

        svals = torch.linalg.svdvals(J)  # (N, C)
        smin = svals[..., -1]  # (N,)
        cost = torch.nn.functional.softplus(self.margin_sigma_min - smin).mean()
        return self.reg_weight * cost

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
        self.input = nn.Linear(self.obs_dim, config.d_model, bias=True, device=config.device)
        self.ssm = SSM(config, device=config.device)
        self.head = torch.nn.Parameter(
            torch.tensor([0.], device=config.device))  # nn.Linear(N_UNITS, 1, bias=False, device=DEVICE)    # scalar Q
        # temperature parameter (log parameterization for positivity)
        self.log_tau = nn.Parameter(torch.zeros((1), device=config.device))
        # self.sa = torch.tensor([0.], device=config.device)
        # self.sa_net = torch.nn.Linear(in_features=self.ssm.args.d_inner * self.ssm.args.d_state, out_features=config)
        self.hidden = None  # b, p, h, s

        # Inference cache for sequential rollout
        self.sequential = False
        self.cache = None

    # ---------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, *, k: int):
        """obs: (T, B*k, obs_dim)   → values (T,B,k) and logits (T,B,k)"""
        T, Bk, _ = obs.shape

        # if operating with self action need to add action to network.
        if self.sequential and self.cache is None:
            self.cache = InferenceCache.alloc(batch_size=Bk, args=self.config, device=self.config.device)
        if self.sequential:
            h = self.cache
        else:
            h = None
        x = (self.input(obs.float().view(T * Bk, -1)))  # (T*Bk,N_UNITS)
        x = rearrange(x, '(t b) s -> t b s', t=T, b=Bk)  # (T,Bk,N_UNITS)

        y, self.cache = self.ssm(x.transpose(0, 1), h)  # (T,Bk,N_UNITS)
        self.hidden = self.cache.ssm_state.clone()
        y = y.transpose(0, 1)

        q = torch.sum(y, dim=-1) * torch.nn.functional.softplus(self.head)  # (T,Bk)
        q = q.view(T, -1, k)  # (T,B,k)

        tau = torch.exp(self.log_tau)
        logits = q / tau
        return q, logits

    # ---------------------------------------------------------------------
    def reset(self):
        self.sequential = False
        self.cache = None
