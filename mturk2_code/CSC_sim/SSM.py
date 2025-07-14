import torch
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as F
from torch.fx.experimental.partitioner_utils import Device


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.
    """
    T = x.size(-1)
    x = repeat(x, "... t -> ... t e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x: Tensor, A: Tensor, B: Tensor, C: Tensor, chunk_size: int, initial_states: Tensor = None,
        device: str = "cpu"):
    """Structed State Space Duality
    :param x: (t, b, s, p)
    :param A: (t, b, s)
    :param B: (t, b, s, n)
    :param C: (t, b, s, n)
    """
    assert x.shape[0] % chunk_size == 0

    x = x.transpose(0, 1)  # (b, t, s, p)
    A = A.transpose(0, 1)  # (b, t, s)
    B = B.transpose(0, 1)  # (b, t, s, n)
    C = C.transpose(0, 1)  # (b, t, s, n)

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l s -> b s c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # recurent connection between same chunl
    L = torch.exp(segsum(A, device=device)) # <b s t l l=m>
    Y_diag = torch.einsum("bclsn, bcmsn, bsclm, bcmsp -> bclsp", C, B, L, x)

    # intra-chunk state
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclsn, bscl, bclsp -> bcspn", B, decay_states, x)

    # between chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bszc, bcspn -> bzspn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # chunk readout
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclsn, bcspn, bscl -> bclsp", C, states, state_decay_out)

    # combined intra and inter chunk terms.
    Y = rearrange(Y_diag + Y_off, "b c l s p -> b (c l) s p")  # <b, t, s, p>

    return Y.transpose(0, 1), final_state


class SSD_cell(torch.nn.Module):
    """
    A basic Mamba-2 implimentation designed to be configurable
    batch size: B
    """

    def __init__(self, spatial=10, in_channels=10, head_dim=10, state_dim=8, conv_size=5, chunk_size=64, device="cpu",
                 *args, **kwargs):
        """
        :param spatial: Num parallel spatial heads (number of cell heads).
        :param in_channels: input feature channels (the model dimension)
        :param head_dim: Dimmensionality of each spatial head.
        :param state_dim: number of recurrent states.
        :param out_channels: output feature channels.
        :param conv_size: kernel size for parameter compute (B, C)
        :param chunk_size: how to chunk an input sequence in parallel mode. Optimal setting depends on GPU memory and
                           model dimension. Determines both sequence chunking for conv and SSD matrix block size.
        """

        super().__init__(*args, **kwargs)
        self.bc_conv = torch.nn.Conv1d(in_channels, 2 * state_dim + head_dim + 1, kernel_size=conv_size, device=device) # position one is the logit, 2 is value est. The spatial dimension will be collapsed into batch when this is applied
        self.passthrough = torch.nn.Linear(in_channels, 1, device=device)
        self.reduce = torch.nn.Linear(head_dim, 1, device=device)
        self.dt_param = torch.nn.Parameter(torch.tensor([1.]))
        self.dt_bias = .01

        # the input to the conv in conv_size previous state + the cells output state of each channel.
        self.A = torch.zeros(head_dim, device=device) # we're gonna fix dt at some learned value that can control the timescale of a population and allow A to varying with input to control evidence accumulation

        self.chunk_size = chunk_size
        self.device = device
        self.context_size = conv_size
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.head_dim = head_dim
        self.spatial = spatial
        self.context = []
        self.activ = torch.nn.Sigmoid()
        self.hidden = torch.zeros(1, spatial, state_dim, head_dim, device=device) # t b s n p

        # saved A, B, C and X for parallel recompute.
        self.As = []
        self.Bs = []
        self.Cs = []
        self.x2s = []

    def forward(self, x, save_params=False):
        """

        :param x: <time, batch, spatial, features>
        :param save_params: whether to save intermediate A, B, C, and x when running in recurrent mode
        :return y
        """
        seq_len = x.shape[0]
        if seq_len == 1:
            y = self.recurrent_step(x, save_params=save_params)
        else:
            y = self.parallel_step(x)
        return y

    def _compute_ssm_params(self, x, last_only=False):
        """
        :param x: <time, batch, spatial, features>
        :param last_only: whether to compute last state only (e.g for recurrent mode, will pad to context size or cut to context size)
        :return:
        """
        in_seq_len = x.shape[0]
        if last_only:
            if x.shape[0] >= self.context_size:
                x = x[-self.context_size:]
                cont_pad = 0
            else: #last_only  and x.shape[0] < self.context_size:
                cont_pad = self.context_size - x.shape[0]
            out_seq_len = 1
        else:
            cont_pad = self.context_size - 1
            out_seq_len = in_seq_len # this
        batch_size = x.shape[1]
        x = x.reshape(in_seq_len, -1, self.in_channels)
        x = x.permute(1, 2, 0)
        xc = torch.nn.functional.pad(x, (cont_pad, 0), mode='constant', value=0.)
        bcax = self.bc_conv(xc)  # this will be chunked for more efficiency
        bcax = torch.sigmoid(bcax) + bcax # add some nonlinearity.
        bcax = bcax.view(-1, self.spatial, 2 * self.state_dim + self.head_dim + 1, out_seq_len)
        B = bcax[..., :self.state_dim, :]  # <b, s, n, t>
        C = bcax[..., self.state_dim:2*self.state_dim, :] # <b, s, n, t>
        A = bcax[..., 2*self.state_dim, :] # <b, s, t>
        xc = bcax[..., 2*self.state_dim + 1:, :] # <b, s, p, t>

        # combined A with the default time constant (log Ad) and get discrete B
        dt = F.softplus(self.dt_param) + self.dt_bias  # <1,>
        lAd = dt * A # <b, s, t>
        Bd = B * ((torch.exp(lAd) - 1) / A).unsqueeze(2)

        # compute D
        x2 = x[..., -out_seq_len:] # get only the needed instances
        x2 = x2.permute(2, 0, 1).reshape(-1, self.in_channels)
        Dx = self.passthrough(x2)
        Dx = Dx.reshape(out_seq_len, batch_size, self.spatial)
        Bd = Bd.movedim(3, 0)
        C = C.movedim(3, 0)
        lAd =lAd.movedim(2, 0)
        xc = xc.movedim(3, 0)
        return xc, lAd, Bd, C, Dx

    def recurrent_step(self, x, save_params=False):
        """
        :param x: <1, batch, spatial, features>
        :param save_params: whether to save intermediate A, B, C, and x when running
        :return: Tensor y <1, batch, spatial>
        """
        self.context.append(x)
        if len(self.context) > self.context_size:
            self.context.pop(0)
        cur_context = torch.concatenate(self.context, dim=0) # <context, batch, spatial, features>
        x, A, B, C, D = self._compute_ssm_params(cur_context, last_only=True)
        if save_params:
            self.As.append(A)
            self.Bs.append(B)
            self.Cs.append(C)
            self.Ds.append(D)
        Bx = torch.einsum('tbsn,tbsp->tbsnp', B, x)
        self.hidden = (torch.exp(A[..., None, None]) * self.hidden.clone().unsqueeze(0) + Bx)[0]  # <batch, spatial, state_dim, head_dim>
        y_f = torch.einsum('tbsn,bsnp->tbsp', C, self.hidden).reshape(-1, self.head_dim) # <batch * spatial, head_dim>
        y = self.reduce(y_f).reshape(1, -1, self.spatial) # <time, batch, spatial>
        y = y + self.activ(D)
        return y

    def parallel_step(self, x):
        """
        :param x: <time, batch, spatial, features>
        :return: Tensor y <time, batch, spatial>
        """
        # Can impliment sequenctiol chunking for the low memory case.
        x, A, B, C, D = self._compute_ssm_params(x)
        # start chunk
        y_f, hidden = ssd(x, A, B, C, chunk_size=self.chunk_size, device=self.device,)
        # set internal state so we can continue with another pass
        self.hidden = hidden
        y = self.reduce(y_f.reshape((-1, self.head_dim))).reshape(x.shape[0], -1, self.spatial) # <t, b, s>
        y = y + self.activ(D)
        return y

    def reset(self):
        """ reset any saved states to defualt"""
        self.As, self.Bs, self.Cs, self.x2s, self.context = [], [], [], [], []
        self.hidden = torch.zeros(1, self.spatial, self.state_dim, self.head_dim, device=self.device)