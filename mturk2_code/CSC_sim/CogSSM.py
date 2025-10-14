from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat

Device = str | torch.device | None

# ---------------- config ----------------
@dataclass
class SSMConfig:
    d_model: int = 256
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64

    # existing I/O overrides
    input_dim: int | None = None
    output_dim: int | None = None

    # reward feature size for B(reward)
    reward_dim: int = 1

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.input_dim is None:
            self.input_dim = self.d_model
        if self.output_dim is None:
            self.output_dim = self.d_model


# --------------- cache ------------------
class InferenceCache:
    # (batch, d_inner + d_state, d_conv)  # CHANGED (was d_inner + 2*d_state)
    conv_state: Tensor
    # (batch, nheads, headdim, d_state)
    ssm_state: Tensor

    def __init__(self, conv_state: Tensor, ssm_state: Tensor):
        self.conv_state = conv_state
        self.ssm_state = ssm_state

    @staticmethod
    def alloc(batch_size: int, args: SSMConfig, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )

    def clone(self):
        import copy
        return copy.deepcopy(self)


# --------------- model ------------------
class SSM(nn.Module):
    def __init__(self, args: SSMConfig, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # in-proj now produces: z (d_inner), xC preconv (d_inner + d_state), dt (nheads)
        d_in_proj = 2 * args.d_inner + 1 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.input_dim, d_in_proj, bias=True, device=device)

        # depthwise conv over xC only
        self.conv_dim = args.d_inner + args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.d_conv,
            groups=self.conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        # B is linear in reward ONLY
        self.B_proj = nn.Linear(args.reward_dim, args.d_state, bias=True, device=device)

        self.dt_bias = nn.Parameter(torch.zeros(args.nheads, device=device))
        self.A_log = torch.zeros(args.nheads, device=device)
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.output_dim, bias=False, device=device)

    # ---- Shared helpers -----------------------------------------------------

    def _project(self, u: Tensor, reward: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        u: (b, l, input_dim)
        Returns:
            z:   (b, l, d_inner)
            xC:  (b, l, d_inner + d_state)   # pre-conv features for x and C
            dt:  (b, l, nheads), softplus(+bias) applied
            B:  (b, l, state_dim)
        """
        zxc_dt = self.in_proj(u)  # (b, l, 2*d_inner + d_state + nheads)
        z, xC, dt = torch.split(
            zxc_dt,
            [self.args.d_inner, self.conv_dim, self.args.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        # compute B from reward only
        l = reward.shape[1]
        B = self.B_proj(rearrange(reward, "b l 1 -> (b l) 1"))                   # (b, l, d_state)
        B = rearrange(B, "(b l) s -> b l s", l=l)

        return z, xC, dt, B

    def _conv_and_split(
        self, xC: Tensor, h: InferenceCache | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Depthwise conv over xC, split into x and C.

        Args:
            xC: (b, l, d_inner + d_state)
            h : None (full) or cache (step, l==1)

        Returns:
            x:          (b, l, d_inner)
            C:          (b, l, d_state)
            conv_state: (b, d_inner + d_state, d_conv)
        """
        if h is None:
            l = xC.shape[1]
            xC_conv = silu(self.conv1d(xC.transpose(1, 2)).transpose(1, 2))[:, :l, :]
            conv_state = F.pad(rearrange(xC, "b l d -> b d l"), (self.args.d_conv - l, 0))
        else:
            assert xC.shape[1] == 1, "Step mode expects seqlen=1"
            xC_last = xC.squeeze(1)  # (b, conv_dim)
            # roll buffer and insert
            h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
            h.conv_state[:, :, -1] = xC_last
            # depthwise conv
            weight_dw = rearrange(self.conv1d.weight, "d 1 w -> d w")
            xC_conv = torch.sum(h.conv_state * weight_dw, dim=-1)
            xC_conv = xC_conv + self.conv1d.bias
            xC_conv = silu(xC_conv).unsqueeze(1)
            conv_state = h.conv_state

        x, C = torch.split(xC_conv, [self.args.d_inner, self.args.d_state], dim=-1)
        C = torch.ones_like(C)
        return x, C, conv_state

    # ---- Public API ---------------------------------------------------------

    def forward(self, u: Tensor, reward: Tensor, h: InferenceCache | None = None):
        """
        u:       (batch, seqlen, input_dim)
        reward:  (batch, seqlen, reward_dim)   -> ONLY used to compute B
        Returns:
            y: (batch, seqlen, output_dim)
            h: InferenceCache
        """
        if h is not None:
            return self.step(u, reward, h)

        # projections
        A = -torch.exp(self.A_log)                # (nheads,)
        z, xC, dt, B = self._project(u, reward)              # (b,l,d_inner), (b,l,conv_dim), (b,l,nheads)
        x, C, conv_state = self._conv_and_split(xC, h=None)
        # SSD
        x_heads = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x_heads * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),     # broadcast over heads
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        new_h = InferenceCache(conv_state, ssm_state)
        return y, new_h

    def step(self, u: Tensor, reward: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """
        Single-token step.
        u:      (batch, 1, input_dim)
        reward: (batch, 1, reward_dim)
        Returns:
            y: (batch, 1, output_dim)
        """
        assert u.shape[1] == 1 and reward.shape[1] == 1, "step() expects seqlen=1"

        bs = u.shape[0]
        if bs > 1 and h.conv_state.shape[0] == 1 and h.ssm_state.shape[0] == 1:
            h.conv_state = torch.tile(h.conv_state, (bs, 1, 1))
            h.ssm_state = torch.tile(h.ssm_state, (bs, 1, 1, 1))

        # shared projections
        z, xC, dt, B = self._project(u, reward)             # (b,1,·)
        x, C, _ = self._conv_and_split(xC, h)

        # B from reward only
        B = self.B_proj(reward).squeeze(1)       # (b, d_state)

        # scalar SSM update
        A = -torch.exp(self.A_log)
        dt = dt.squeeze(1)                       # (b, nheads)
        dA = torch.exp(dt * A)
        x = rearrange(x.squeeze(1), "b (h p) -> b h p", p=self.args.headdim)
        C = C.squeeze(1)                         # (b, d_state)

        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z.squeeze(1))
        y = self.out_proj(y)

        return y.unsqueeze(1), h


# --------- utilities (unchanged) ----------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x  # * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    return F.sigmoid(x) + .1*x


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78

     c -> t
     l -> l
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader
    x, A, B, C = [
        rearrange(m, "b (t l) ... -> b t l ...", l=chunk_size) for m in (x, A, B, C) # let u be a second time axis
    ]

    A = rearrange(A, "b t l h -> b h t l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state