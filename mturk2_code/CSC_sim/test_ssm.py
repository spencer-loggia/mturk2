import math
import importlib
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