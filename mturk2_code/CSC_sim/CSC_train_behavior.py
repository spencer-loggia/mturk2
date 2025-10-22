import torch
from CSC_train_value import *
import torch.nn.functional as F
from ColorShapeSpace_behave import make_csc2_behavior_vector_from_csvs

def train_behavior(csv_paths, tau: float = 1.0):
    # --- Build vectorized behavior env (one CSV per env) ---
    vec = make_csc2_behavior_vector_from_csvs(csv_paths)
    B, T, k = vec.num_envs, vec.envs[0].T, vec.envs[0].k
    assert T == MAX_TRIALS, f"Expected MAX_TRIALS={MAX_TRIALS}, got {T}"
    assert all(e.T == T and e.k == k for e in vec.envs), "All episodes must share T and k"

    # --- Model / optim ---
    agent = QAgent(agent_config).to(DEVICE)
    deform = Deform(channels=agent_config.percept_dim)
    optimizer = torch.optim.Adam(list(agent.parameters()) + list(deform.parameters()), lr=LR)

    loss_hist, acc_hist = [], []

    # Stack dataset tensors once: angles (T,B,k,2), choices (T,B), rewards (T,B)
    angles  = np.stack([e.angles  for e in vec.envs], axis=1)
    choices = np.stack([e.choices for e in vec.envs], axis=1)
    rewards = np.stack([e.rewards for e in vec.envs], axis=1)

    # prev_r[t] = reward[t-1], zeros at t=0
    prev_r = np.zeros_like(rewards, dtype=np.float32)
    if T > 1: prev_r[1:] = rewards[:-1]

    angles_tbk2 = torch.from_numpy(angles).to(DEVICE)       # (T,B,k,2)
    prev_r_tb   = torch.from_numpy(prev_r).to(DEVICE)       # (T,B)
    target_tb   = torch.from_numpy(choices).to(DEVICE)      # (T,B)

    for epoch in range(1, EPOCHS + 1):
        # --------- Build per-time, per-option features and run SSM in parallel ---------
        TB = T * B
        to_select = torch.randint(0, B, size=(2,))
        oa_tbk2 = angles_tbk2[:, to_select, :].reshape(TB, k, 2)             # (TB,k,2)
        pr_tb   = prev_r_tb[:, to_select].reshape(TB)                      # (TB,)
        x = build_value_obs(oa_tbk2, pr_tb)                  # (TB*k, 2*DIMS+1)

        # Perceptual deformation (+ training-time noise to ground units)
        p = deform(x[:, :agent_config.percept_dim])
        p = p + torch.normal(mean=0., std=.1, size=p.shape, device=p.device)
        x[:, :agent_config.percept_dim] = p
        x_tBkF = x.view(T, B * k, x.shape[-1])               # (T, B*k, F)

        agent.reset()
        q_all, _ = agent(x_tBkF, k=k)                        # (T,B,k) estimated values

        # --------- Behavior loss: CE(onehot(choice), softmax(tau * V)) ---------
        logits = (tau * q_all).reshape(T * B, k)
        target = target_tb[:, to_select].reshape(T * B)
        ce_loss = F.cross_entropy(logits, target)
        loss = ce_loss + deform.compute_penalty()

        # --------- Optimize ---------
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # --------- Logging ---------
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).float().mean().item()
        loss_hist.append(ce_loss.item()); acc_hist.append(acc)

        if epoch % 10 == 0:
            print(f"Ep {epoch:04d} | CE {ce_loss.item():.3f} | Acc {acc:.2%}")

        if epoch % EVAL_INTERVAL == 0:
            with open(f"models/cog_ssm_{epoch}.pkl","wb") as f: pickle.dump(agent.state_dict(), f)
            with open(f"models/deform_{epoch}.pkl","wb") as f:  pickle.dump(deform.state_dict(), f)
            print("--- Simulated Evaluation ---")
            play_once(agent, deform, trials=MAX_TRIALS, render=False, k=2)

    print("=== Final (Simulated) Policy Evaluation ===")
    play_once(agent, deform, trials=MAX_TRIALS, render=False)

    # Optional: quick curves
    if len(loss_hist) >= 25:
        w = 25; filt = np.ones(w)/w
        loss_sm = np.convolve(loss_hist, filt, mode='valid')
        acc_sm  = np.convolve(acc_hist,  filt, mode='valid')
        plt.figure(figsize=(8,4))
        plt.plot(loss_sm, label='CE Loss'); plt.plot(acc_sm, label='Choice Acc')
        plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout()
        plt.title('Behavior Training Curves (window=25)'); plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

