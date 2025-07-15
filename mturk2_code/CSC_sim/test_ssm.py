# tests/test_ssm.py
import math
import torch
import pytest
import SSM as ssm

segsum = ssm.segsum
ssd = ssm.ssd
SSMConfig = ssm.SSMConfig
SSM = ssm.SSM
InferenceCache = ssm.InferenceCache
from torch.nn import MSELoss
from torch.optim import Adam
import SSM as ssm
import numpy as np

# ------------------------------------------------------------------
# segsum ------------------------------------------------------------------
def test_segsum_structure_and_values():
    torch.manual_seed(0)
    x = torch.randn(5)
    S = segsum(x)
    # shape
    assert S.shape == (5, 5)
    # diag / upper-triangular pattern
    for i in range(5):
        assert S[i, i] == 0
        assert torch.all(S[i, i + 1 :] == -math.inf)
        assert torch.all(torch.isfinite(S[i, : i]))


def test_segsum_cumulants():
    torch.manual_seed(1)
    x = torch.randn(4)
    S = segsum(x)
    for i in range(1, 4):
        for j in range(i):
            expected = x[j + 1 : i + 1].sum()
            assert torch.allclose(S[i, j], expected, atol=1e-6)

# ------------------------------------------------------------------
# ssd ------------------------------------------------------------------
def test_ssd_shapes_and_numerics():
    torch.manual_seed(42)
    B, T, H, P, N = 2, 8, 3, 4, 5          # H = n_heads, P = d_head
    X  = torch.randn(B, T, H, P)
    A  = torch.randn(B, T, H)
    Bm = torch.randn(B, T, H, N)
    Cm = torch.randn(B, T, H, N)

    Y, final_state = ssd(X, A, Bm, Cm, chunk_size=4)

    assert Y.shape == (B, T, H, P)
    assert final_state.shape == (B, H, P, N)
    assert not torch.isnan(Y).any()
    assert not torch.isnan(final_state).any()


def test_ssd_chunk_divisor_assert():
    torch.manual_seed(0)
    B, T, H, P, N = 1, 7, 1, 1, 2
    X  = torch.randn(B, T, H, P)
    A  = torch.randn(B, T, H)
    Bm = torch.randn(B, T, H, N)
    Cm = torch.randn(B, T, H, N)
    with pytest.raises(AssertionError):
        ssd(X, A, Bm, Cm, chunk_size=4)

# ------------------------------------------------------------------
# SSM end-to-end -----------------------------------------------------
def test_ssm_parallel_vs_step():
    torch.manual_seed(123)
    B, T, D = 4, 12, 8                      # model dim D
    cfg = SSMConfig(d_model=D, d_state=3, d_conv=3, expand=1, headdim=4, chunk_size=4)
    model = SSM(cfg, device="cpu")
    u = torch.randn(B, T, D)

    # full sequence (parallel)
    y_par, _ = model(u)                     # (B,T,D)

    # token-wise (recurrent)
    h = InferenceCache.alloc(B, cfg)
    outs = []
    for t in range(T):
        y_t, h = model.step(u[:, t : t + 1], h)
        outs.append(y_t)
    y_seq = torch.cat(outs, dim=1)

    assert torch.allclose(y_par, y_seq, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------
# 1.  SSM learns to copy the feature-mean of its input
# ------------------------------------------------------------------
def test_ssm_learns_copy_mean():
    torch.manual_seed(0)

    # task ----------------------------------------------------------
    T, B, D = 50, 1, 3                       # seq, batch, feature-dim
    inp = torch.randn(B, T, D)
    target = inp.mean(dim=-1, keepdim=True)  # (B,T,1)

    # model ---------------------------------------------------------
    cfg = ssm.SSMConfig(
        d_model=D, d_state=4, d_conv=1, expand=1, headdim=1, chunk_size=T
    )
    model = ssm.SSM(cfg)
    model.train()

    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = MSELoss()

    # training ------------------------------------------------------
    steps, tol = 300, 1e-2
    for _ in range(steps):
        opt.zero_grad()
        out, _ = model(inp)                  # (B,T,D)
        loss = loss_fn(out.mean(-1, keepdim=True), target)
        loss.backward()
        opt.step()

    # verification --------------------------------------------------
    model.eval()
    with torch.no_grad():
        out, _ = model(inp)
        final = loss_fn(out.mean(-1, keepdim=True), target).item()
    assert final < tol, f"MSE {final:.4f} > {tol}"


# ------------------------------------------------------------------
# 2.  SSM learns one-step ahead prediction of the feature-mean
# ------------------------------------------------------------------
def test_ssm_learns_next_token_mean():
    torch.manual_seed(0)

    # task ----------------------------------------------------------
    T, B, D = 20, 1, 4
    vals = np.array([-0.2, -0.1, 0.1, 0.2], dtype=float)
    seq = torch.from_numpy(np.random.choice(vals, size=T)).float()      # (T,)

    # add axes: (1,T,1) -> broadcast over D
    inp = seq[None, :, None].repeat(B, 1, D)                            # (B,T,D)

    target = torch.zeros(B, T, 1)
    target[:, :-1] = inp[:, 1:].mean(dim=-1, keepdim=True)              # predict next-token mean

    # model ---------------------------------------------------------
    cfg = ssm.SSMConfig(
        d_model=D, d_state=8, d_conv=1, expand=1, headdim=2, chunk_size=T
    )
    model = ssm.SSM(cfg)
    model.train()

    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = MSELoss()

    # training ------------------------------------------------------
    steps, tol = 800, 1e-2
    for _ in range(steps):
        opt.zero_grad()
        out, _ = model(inp)
        loss = loss_fn(out.mean(-1, keepdim=True), target)
        loss.backward()
        opt.step()

    # verification --------------------------------------------------
    model.eval()
    with torch.no_grad():
        out, _ = model(inp)
        final = loss_fn(out.mean(-1, keepdim=True), target).item()
    print()
    print(torch.round(target.squeeze().detach().cpu() * 10).int().tolist())
    print(torch.round(out.mean(-1, keepdim=True).squeeze().detach().cpu() * 10).int().tolist())
    assert final < tol, f"MSE {final:.4f} > {tol}"