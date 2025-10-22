import torch

from train_utilities import *
from ColorShapeSpace_behave import make_csc2_behavior_vector_from_csvs
from neurotools.util import is_converged
from loss import TemporalDifference
from matplotlib import pyplot as plt

# ================================ Training ===================================

def train(cfg: TrainConfig):
    agent, deform = QAgent(cfg).to(cfg.device), Deform(channels=cfg.percept_dim, deform_basis=4, groups=cfg.dims, device=cfg.device)
    td_loss_fn = TemporalDifference(gamma=cfg.gamma, normalize=True)
    opt = torch.optim.Adam(list(agent.parameters()) + list(deform.parameters()), lr=cfg.lr)

    loss_hist, ret_hist, storage = [], [], {}
    one_d = (cfg.dims == 1)

    for epoch in range(1, cfg.epochs + 1):
        agent.reset()
        n = 72 if cfg.dims == 1 else 36
        vec = make_csc2_vector(cfg.batch_size, n=n, batched=False, trials=cfg.max_trials,
                               k=cfg.num_choices, one_d=one_d,
                               seeds=[cfg.seed_override] * cfg.batch_size)
        obs_seq, rewards, _ = collect_batch_q(vec, agent, deform, cfg)

        # --- simple replay (kept to preserve original training dynamics) ---
        keys = random.sample(list(storage), min(len(storage), 5 * cfg.batch_size))
        r_obs = torch.concatenate([obs_seq] + [storage[k_][0] for k_ in keys], dim=1) if keys else obs_seq
        r_rew = torch.concatenate([rewards] + [storage[k_][1] for k_ in keys], dim=1) if keys else rewards

        q_all, _ = agent(r_obs, r_rew[:-1], k=1)                      # (T,B,1)
        q_pred = q_all.squeeze(-1)                        # (T,B)

        td = td_loss_fn(q_pred, r_rew[1:])
        loss = td + deform.compute_penalty()

        opt.zero_grad(); loss.backward(); opt.step()
        opt, _ = is_converged(loss_hist, opt, cfg.batch_size, epoch, max_lr=.01)

        # maintain replay buffer size
        drop = random.sample(list(storage), max(0, len(storage) - 20 * cfg.batch_size))
        storage = {k_: storage.pop(k_) for k_ in drop}
        for b in range(cfg.batch_size):
            storage[f"{epoch}:{b}"] = (obs_seq[:, b:b+1].detach(), rewards[:, b:b+1].detach())

        loss_hist.append(td.item()); ret_hist.append(rewards[-100:].mean().item())
        if epoch % 10 == 0:
            print(f"Ep {epoch:04d} | TD {td.item():.3f} | R {ret_hist[-1]:.2f}")

        if epoch % cfg.eval_interval == 0:
            save_checkpoint(cfg, agent, deform, epoch, tag="value")
            print("--- Evaluation ---")
            play_once(agent, deform, cfg, trials=cfg.max_trials, render=False, k=2)

    print("=== Final Policy Evaluation ===")
    play_once(agent, deform, cfg, trials=cfg.max_trials, render=False)

    if len(loss_hist) >= 25:
        w = 25; filt = np.ones(w) / w
        loss_sm = np.convolve(loss_hist, filt, mode='valid')
        ret_sm  = np.convolve(ret_hist,  filt, mode='valid')
        plt.figure(figsize=(8,4))
        plt.plot(loss_sm, label='TD Loss'); plt.plot(ret_sm, label='Mean Return')
        plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout()
        plt.title('Training Curves (window=25)'); plt.show()

# ============================= Behavior training =============================

# If you’re using the deterministic CSV behavior env, uncomment the import above
# and use this trainer. No replay per your earlier guidance.

def train_behavior(cfg: TrainConfig, csv_paths: Sequence[str]):
    from ColorShapeSpace_behave import make_csc2_behavior_vector_from_csvs
    vec = make_csc2_behavior_vector_from_csvs(csv_paths, max_trials=cfg.max_trials)
    tau = torch.nn.Parameter(torch.tensor(cfg.temperature, device=cfg.device))
    td_loss_fn = TemporalDifference(gamma=cfg.gamma, normalize=True)

    B, T, k = 2, vec.envs[0].T, vec.envs[0].k
    assert T == cfg.max_trials, f"Expected max_trials={cfg.max_trials}, got {T}"
    assert all(e.T == T and e.k == k for e in vec.envs), "All episodes must share T and k"

    agent, deform = QAgent(cfg).to(cfg.device), Deform(channels=cfg.percept_dim, deform_basis=4, groups=cfg.dims, device=cfg.device)
    opt = torch.optim.Adam(list(agent.parameters()) + list(deform.parameters()) + [tau], lr=cfg.lr)

    loss_hist, acc_hist = [], []

    # Stack static dataset once
    angles  = np.stack([e.angles  for e in vec.envs], axis=1)  # (T,B,k,2)
    choices = np.stack([e.choices for e in vec.envs], axis=1)  # (T,B)
    rewards = np.stack([e.rewards for e in vec.envs], axis=1)  # (T,B)
    rewards = (rewards - rewards.mean()) / rewards.std()

    prev_r = np.zeros((T, B), dtype=np.float32)

    angles_tbk2 = torch.from_numpy(angles).to(cfg.device)
    target_tb   = torch.from_numpy(choices).to(cfg.device)

    for epoch in range(1, cfg.epochs + 1):
        TB = T * B
        sample = tuple(random.randint(0, vec.num_envs - 1) for _ in range(B))
        oa_tbk2 = angles_tbk2[:, sample, :].reshape(TB, k, 2)
        prev_r[1:] = rewards[:-1, sample]
        prev_r_tb = torch.from_numpy(prev_r).to(cfg.device)
        pr_tb   = torch.tile(prev_r_tb.reshape(TB, 1), (1, k))
        x = build_value_obs(oa_tbk2, cfg)                          # (TB*k, 2*cfg.dims)
        n_x = x.clone()
        n_x = apply_deform_inplace(n_x, deform, cfg, add_noise=True)
        x_tBkF = n_x.view(T, B * k, x.shape[-1])
        pr_tb = pr_tb.view(T, B * k, 1)
        agent.reset()
        q_all, _ = agent(x_tBkF, pr_tb, k=k)                                     # (T,B,k)
        q_pred = q_all[np.arange(T)[:, None], np.arange(B)[None, :], target_tb[:, sample]]

        logits = (tau * q_all).reshape(T * B, k)
        target = target_tb[:, sample].reshape(T * B)
        ce = F.cross_entropy(logits, target)

        # compute auxillary TD loss
        td = td_loss_fn(q_pred[-10000:], torch.from_numpy(rewards[-10000:, sample]).to(cfg.device))

        loss = ce + deform.compute_penalty() + (1 / 100000) * td

        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (logits.argmax(-1) == target).float().mean().item()
        loss_hist.append(ce.item()); acc_hist.append(acc)
        opt, _ = is_converged(loss_hist, opt, cfg.batch_size, epoch, max_lr=.01)

        if epoch % 10 == 0:
            print(f"Ep {epoch:04d} | CE {ce.item():.3f} | Acc {acc:.2%}| TD {(1 / 100000) * td.detach().cpu().item():.2f}")

        if epoch % cfg.eval_interval == 0:
            save_checkpoint(cfg, agent, deform, epoch, tag="behavior")
            print("--- Simulated Evaluation ---")
            play_once(agent, deform, cfg, trials=cfg.max_trials, render=False, k=2)

    cfg.temperature = tau.detach().cpu().item()
    print("=== Final (Simulated) Policy Evaluation ===")
    play_once(agent, deform, cfg, trials=cfg.max_trials, render=False)

    if len(loss_hist) >= 25:
        w = 25; filt = np.ones(w) / w
        loss_sm = np.convolve(loss_hist, filt, mode='valid'); acc_sm = np.convolve(acc_hist, filt, mode='valid')
        plt.figure(figsize=(8,4))
        plt.plot(loss_sm, label='CE Loss'); plt.plot(acc_sm, label='Choice Acc')
        plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout(); plt.title('Behavior Curves (window=25)'); plt.show()

# ================================ Entrypoint =================================

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # Example config mirroring your previous defaults
    cfg = TrainConfig(
        d_model=9, d_state=1, d_conv=2, expand=1,
        percept_dim=4, reward_dim=1, action_dims=(1,), device="cuda",
        headdim=9, chunk_size=512,
        batch_size=10, epochs=5000, eval_interval=1000,
        lr=2e-2, gamma=0.0, dims=2, max_trials=512*150, num_choices=1,
        seed_override=None, run_name="cogssm",
        input_dim=4, output_dim=1,
    )
    # train(cfg)
    # For behavior learning:
    train_behavior(cfg, ["saved_data/all_Sally.csv",
                                  "saved_data/all_Buzz.csv"])