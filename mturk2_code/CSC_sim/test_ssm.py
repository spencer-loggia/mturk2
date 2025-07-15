import math
import importlib

import numpy as np
import pytest
import torch
import SSM as ssm

# Handy shortcuts
segsum = ssm.segsum
ssd     = ssm.ssd
SSDCell = ssm.SSD_cell


# -----------------------------------------------------------------
# 1.  segsum: structure check
# -----------------------------------------------------------------
def test_segsum_triangular():
    torch.manual_seed(0)
    x = torch.randn(5)
    S = segsum(x)

    # shape
    assert S.shape == (5, 5)

    # strict upper triangle = −∞, diag = 0
    for i in range(5):
        assert torch.isclose(S[i, i], torch.tensor(0.0))            # diag
        for j in range(i + 1, 5):
            assert S[i, j].item() == -math.inf                      # above diag
        for j in range(i):
            assert torch.isfinite(S[i, j])                          # below diag


# -----------------------------------------------------------------
# 2.  segsum: numerical content check (lower-triangular part)
#     S[i, j] should equal sum_{k=j+1}^{i} x[k]   (i > j)
# -----------------------------------------------------------------
def test_segsum_cumulative():
    torch.manual_seed(1)
    x = torch.randn(4)
    S = segsum(x)

    for i in range(1, 4):
        for j in range(i):
            expected = x[j + 1 : i + 1].sum()
            assert torch.isclose(S[i, j], expected, atol=1e-6)


# -----------------------------------------------------------------
# 3.  ssd: shape & NaN-safety for a small random batch
# -----------------------------------------------------------------
def test_ssd_shapes():
    torch.manual_seed(42)
    B, T, S, P, N = 2, 8, 3, 4, 5
    X = torch.randn(T, B,  S, P)
    A = torch.randn(T, B, S)
    Bm = torch.randn(T, B, S, N)
    Cm = torch.randn(T, B, S, N)

    Y, final_state = ssd(X, A, Bm, Cm, chunk_size=4)

    assert Y.shape == (T, B, S, P)
    assert final_state.shape == (B, S, P, N)
    assert torch.isnan(Y).sum().item() == 0
    assert torch.isnan(final_state).sum().item() == 0


# -----------------------------------------------------------------
# 4.  ssd: chunk-size must divide the sequence length
# -----------------------------------------------------------------
def test_ssd_chunk_assert():
    torch.manual_seed(0)
    B, T, S, P, N = 1, 7, 1, 1, 2
    X = torch.randn(B, T, S, P)
    A = torch.randn(B, T, S)
    Bm = torch.randn(B, T, S, N)
    Cm = torch.randn(B, T, S, N)

    with pytest.raises(AssertionError):
        ssd(X, A, Bm, Cm, chunk_size=4)          # 7 not divisible by 4


# -----------------------------------------------------------------
# 5.  SSD_cell: parallel vs. recurrent consistency on a short seq
# -----------------------------------------------------------------
def test_cell_parallel_equals_sequential():
    torch.manual_seed(123)
    # Small dimensions so the test is lightning-fast
    seq_len, batch, spatial, feat = 12, 4, 2, 6
    cell_cfg = dict(
        spatial=spatial,
        in_channels=feat,
        head_dim=4,
        state_dim=3,
        conv_size=4,          # conv-kernel = 1 ⇒ no context buffering hassle
        chunk_size=4,
        device="cpu",
    )

    cell = SSDCell(**cell_cfg)
    X = torch.randn(seq_len, batch, spatial, feat)

    # --- parallel pass --------------------------------------------------
    Y_parallel = cell(X)                       # (T,B,S)

    # --- sequential pass ------------------------------------------------
    cell.reset()                              # clear hidden/context
    outputs = []
    for t in range(seq_len):
        y_t = cell(X[t : t + 1])              # one-step recurrent
        outputs.append(y_t)
    Y_sequential = torch.cat(outputs, dim=0)   # (T,B,S)

    # close agreement, tolerating minor FP noise
    assert torch.allclose(Y_parallel, Y_sequential, atol=1e-3, rtol=1e-3)


# A short test of sequence copying

def test_ssd_cell_learns_copy_task():
    from torch.nn import MSELoss
    from torch.optim import Adam
    # reproducibility
    torch.manual_seed(0)

    # hyperparameters
    seq_len, batch, spatial, features = 50, 1, 2, 3
    head_dim, state_dim = 4, 2
    chunk_size = seq_len  # no chunking for simplicity
    lr = 1e-2
    steps = 200
    tol = 1e-2

    # model
    model = SSDCell(
        spatial=spatial,
        in_channels=features,
        head_dim=head_dim,
        state_dim=state_dim,
        conv_size=1,
        chunk_size=chunk_size,
        device="cpu"
    )
    model.train()

    # optimizer & loss
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # create a random “sequence” and set target = input projected to output dims
    inp = torch.randn(seq_len, batch, spatial, features)
    # SSD_cell outputs shape (t, b, s); here we’ll copy the L2 norm per feature
    target = inp.mean(dim=-1)  # shape: (t, b, spatial)

    # training loop
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(inp)               # (t, b, s)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        final_out = model(inp)
        final_loss = loss_fn(final_out, target).item()
    print()
    print(target[:, 0, 0].detach().cpu().tolist())
    print(out[:, 0, 0].detach().cpu().tolist())
    # assert that it has learned to “copy”
    assert final_loss < tol, f"Final MSE {final_loss:.4f} exceeds tolerance {tol}"


def test_ssd_cell_learns_next_token_prediction():
    import torch
    from torch.nn import MSELoss
    from torch.optim import Adam
    from SSM import SSD_cell

    # reproducibility
    torch.manual_seed(0)

    # hyperparameters
    seq_len, batch, spatial, features = 20, 1, 2, 3
    head_dim, state_dim = 4, 2
    chunk_size = seq_len  # no chunking
    lr = 1e-2
    steps = 1000
    tol = 1e-3

    # model
    model = SSD_cell(
        spatial=spatial,
        in_channels=features,
        head_dim=head_dim,
        state_dim=state_dim,
        conv_size=1,
        chunk_size=chunk_size,
        device="cpu"
    )
    model.train()

    # optimizer & loss
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    # random sequence; next-token targets
    seq = torch.from_numpy(np.random.choice(np.array([-.2, -.1, .1, .2], dtype=float), size=seq_len))
    inp = torch.zeros(seq_len, batch, spatial, features)
    inp += seq[:, None, None, None]
    target = torch.zeros(seq_len, batch, spatial)
    # for t in [0..seq_len-2], predict the next input’s feature-mean
    target[:-1] = inp[1:].mean(dim=-1)
    # target[-1] stays zero

    # training loop
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(inp)  # (t, b, s)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        final_out = model(inp)
        final_loss = loss_fn(final_out, target).item()
    print()
    print(torch.round(target[:, 0, 0].detach().cpu() * 10).int().tolist())
    print(torch.round(final_out[:, 0, 0].detach().cpu() * 10).int().tolist())
    assert final_loss < tol, f"Final MSE {final_loss:.4f} exceeds tolerance {tol}"