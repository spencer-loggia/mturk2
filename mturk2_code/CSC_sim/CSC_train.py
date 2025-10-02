import pickle
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.vector import SyncVectorEnv
from einops import rearrange

from ColorShapeSpace_sim import CSC2Env, make_csc2_vector

from loss import ActorCritic
from CogSSM import SSM, SSMConfig, InferenceCache
from neurotools.util import is_converged
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# --------------------------- Hyper-parameters ---------------------------- #
ENV_DIM        = 72          # 1D circle resolution
BATCH_SIZE     = 15          # number of parallel environments
EPOCHS         = 1500
EVAL_INTERVAL  = 700         # render every N epochs
CHUNK_SIZE     = 64
GAMMA          = 0.95
DISP_STEPS     = 30
MAX_TRIALS     = CHUNK_SIZE * 15
EVAL_TRIALS    = DISP_STEPS * 32  # trials when rendering policy
ALPHA_ENTROPY  = 0.1
LR             = 0.01
N_UNITS        = 2
HEAD_DIM       = 2
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------- 1D specifics (ported from Q-learning version) ------------- #
DIMS = 1  # number of angular dims (=1 for 1D circle)

def build_policy_obs(obs_a: np.ndarray) -> torch.Tensor:
    """
    Convert angle features into flattened per-step policy features.
    obs_a : ndarray (B, k, DIMS)  – angles for each option
    returns: tensor  (B, k * (2*DIMS))  – [sin,cos] per option, flattened
    """
    B, k, _ = obs_a.shape
    feat = torch.from_numpy(obs_a).float().to(DEVICE)       # B,k,1
    feat = torch.cat([torch.sin(feat), torch.cos(feat)], dim=-1)  # B,k,2
    return feat.view(B, k * (2 * DIMS))                     # B, k*2

# ----------------------------- SSM config -------------------------------- #
ssm_configuration = SSMConfig(
    d_model=N_UNITS,
    d_state=1,
    d_conv=2,
    expand=1,
    headdim=HEAD_DIM,
    chunk_size=CHUNK_SIZE,
)

# --------------------------- Agent & ValueNet --------------------------- #
class ActorAgent(nn.Module):
    """Small actor with an SSM core; consumes flattened [sin,cos] + prev reward."""
    def __init__(self, obs_dim, n_actions, n_units=N_UNITS):
        super().__init__()
        self.input = nn.Linear(obs_dim, n_units, device=DEVICE)
        self.model = SSM(ssm_configuration, device=DEVICE)
        self.output = nn.Sequential(
            nn.Linear(n_units, n_units, device=DEVICE), nn.ReLU(),
            nn.Linear(n_units, 8, device=DEVICE), nn.ReLU(),
            nn.Linear(8, n_actions, device=DEVICE)
        )
        self.device = DEVICE
        self.hidden = []
        self.n_actions = n_actions
        self.n_units = n_units
        self.sequential = False
        self.cache = None

    def _action_from_hidden(self, y):
        t, B, _ = y.shape
        self.hidden = y.detach().clone()
        y = y.reshape(t * B, self.n_units)
        y = torch.relu(y)
        action_logit = self.output(y)  # <tb, a>
        return action_logit.reshape(t, B, self.n_actions)

    def forward(self, obs, save_params=False):   # obs: (t, B, obs_dim)
        t, B, _ = obs.shape
        if self.sequential and self.cache is None:
            self.cache = InferenceCache.alloc(batch_size=B, args=ssm_configuration, device=DEVICE)
        h = self.cache if self.sequential else None

        x = torch.relu(self.input(obs.reshape(t * B, -1)))
        x = rearrange(x, '(t B) s -> t B s', t=t, B=B, s=N_UNITS)
        y, h = self.model.forward(x.transpose(0, 1), h)  # <t, b, s>
        y = y.transpose(0, 1)
        self.cache = h
        return self._action_from_hidden(y)

    def recompute(self):
        y = self.model.recompute()
        return self._action_from_hidden(y)

    def reset(self):
        self.sequential = False
        self.cache = None
        self.hidden = []

class ValueNet(nn.Module):
    """Critic: flattened observation (no prev reward) + actor hidden → scalar V."""
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden, device=DEVICE), nn.ReLU(),
            nn.Linear(hidden, hidden, device=DEVICE), nn.ReLU(),
            nn.Linear(hidden, 1, device=DEVICE)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        :param obs: tensor (time, batch, obs_dim)
        """
        t, b, _ = obs.shape
        v = self.net(obs.reshape(t * b, -1))
        return v.reshape(t, b)

# ---------------------- Trajectory Collection (1D) ---------------------- #
def collect_batch(vec, agent, value_net, chunk_size, device):
    """
    Vectorized rollout over B envs for T=MAX_TRIALS steps.
    Uses 1D features: per-option [sin, cos] + previous reward (scalar).
    """
    B = vec.num_envs
    T_max = MAX_TRIALS
    k = agent.n_actions

    assert T_max % chunk_size == 0, "MAX_TRIALS must be a multiple of CHUNK_SIZE"

    # get initial observations
    obs_a, _ = vec.reset()                    # (B, k, 1) in 1D
    actions = torch.empty(T_max, B, dtype=torch.int64, device=DEVICE)
    rewards = torch.zeros(T_max + 1, B, device=DEVICE)     # reward[t] is from previous step
    obs_batch = torch.zeros(T_max, B, k * (2 * DIMS), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        agent.sequential = True
        for i in range(T_max):
            feat_flat = build_policy_obs(obs_a)            # (B, k*2)
            obs_batch[i] = feat_flat                       # store features (no reward)

            # actor input = features + previous reward scalar
            actor_in = torch.cat([feat_flat.unsqueeze(0), rewards[i][None, :, None]], dim=2)  # (1,B,k*2+1)
            act_logits = agent.forward(actor_in, save_params=True)  # (1,B,k)
            dist = Categorical(logits=act_logits.reshape(B, k))
            actions[i] = dist.sample()

            # env step
            obs_a, reward, _, _, _ = vec.step(actions[i].detach().cpu().numpy())
            rewards[i + 1] = torch.from_numpy(reward).float().to(DEVICE)

    # non-sequential pass for gradients
    agent.reset()
    agent.sequential = False

    # policy logits for all timesteps using stored features + prev reward
    actor_in_all = torch.cat([obs_batch, rewards[:-1].unsqueeze(2)], dim=2)  # (T,B,k*2+1)
    act_logits = agent.forward(actor_in_all)                                  # (T,B,k)

    dist = Categorical(logits=act_logits.reshape(T_max * B, k))
    logp = dist.log_prob(actions.flatten())
    entr = dist.entropy()

    # critic values: concat features (no reward) with actor hidden state
    vals = value_net.forward(torch.cat([obs_batch, agent.hidden], dim=2))
    reward = rewards[1:]  # align with actions

    # reshape back
    logp = logp.reshape(T_max, B)
    entr = entr.reshape(T_max, B)

    # episode returns (monitoring)
    ep_returns = reward[-100:].mean().item()
    return (reward, vals, logp, entr, None, ep_returns)

# --------------------------- Play & Render ------------------------------ #
def play_once(agent, trials: int, disp_steps: int = 30, render=True, seed=None, k=None):
    c = 0
    env = CSC2Env(trials=trials, k=k, n=ENV_DIM, render=render, render_reward=render, one_d=DIMS==1, seed=seed)
    obs, _ = env.reset()  # (k,1)
    total_r = 0.0
    last_r = torch.zeros((1, 1), device=DEVICE)
    agent.reset()
    agent.sequential = True
    states = []

    while True:
        if (c + 1) % DISP_STEPS == 0:
            env.render_reward = True
            env.render_flag = True

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)  # (k,1)
        obs_t = torch.cat([torch.sin(obs_t), torch.cos(obs_t)], dim=-1)  # (k,2)
        obs_t = obs_t.flatten()[None, None, :]                            # (1,1,2k)
        obs_t = torch.cat([obs_t, last_r.unsqueeze(-1)], dim=-1)         # (1,1,2k+1)

        with torch.no_grad():
            logits = agent(obs_t)  # (1,1,k)
        states.append(agent.hidden)
        action = torch.argmax(logits, dim=-1).item()
        obs, r, done, _, _ = env.step(action)

        last_r = torch.tensor(r, device=DEVICE, dtype=torch.float32).reshape((1, 1))
        total_r += float(r)
        c += 1
        env.render_reward = False
        env.render_flag = False
        if done:
            break

    env.close()
    print(f"Rendered episode return: {total_r:.3f}")
    return torch.concatenate(states, dim=0).detach().cpu()

# --------------------------- Training ---------------------------------- #
def train(agent=None, loss_function=None):
    k = 2
    obs_dim = (2 * DIMS) * k  # per-option [sin,cos], flattened

    agent     = ActorAgent(obs_dim + 1, k, N_UNITS).to(DEVICE)  # +1 for prev reward
    value_net = ValueNet(obs_dim + N_UNITS).to(DEVICE)                     # + hidden size
    loss_fn   = ActorCritic(gamma=GAMMA, alpha=ALPHA_ENTROPY)

    act_optim  = torch.optim.Adam(agent.parameters(), lr=LR)
    crit_optim = torch.optim.Adam(value_net.parameters())

    # 1D environment vector
    vec = make_csc2_vector(
        BATCH_SIZE,
        k=k,
        batched=False,
        trials=MAX_TRIALS,
        n=ENV_DIM,
        one_d=True
    )

    critic_hist, actor_hist, return_hist = [], [], []
    for epoch in range(1, EPOCHS + 1):
        agent.reset()
        (rews, vals, logp, entr, mask, ep_ret) = collect_batch(
            vec, agent, value_net, CHUNK_SIZE, DEVICE
        )

        critic_loss, actor_loss = loss_fn(rews, vals, logp, entr, mask)
        loss = critic_loss + actor_loss
        act_optim.zero_grad(); crit_optim.zero_grad(); loss.backward(); act_optim.step(); crit_optim.step()

        # dynamic LR
        act_optim, adone = is_converged(actor_hist, act_optim, BATCH_SIZE, epoch, max_lr=.1)
        crit_optim, cdone = is_converged(critic_hist, crit_optim, BATCH_SIZE, epoch, max_lr=.1)

        critic_hist.append(critic_loss.detach().cpu().item())
        actor_hist.append(actor_loss.detach().cpu().item())
        return_hist.append(ep_ret)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Critic {critic_loss.item():.3f} | Actor {actor_loss.item():.3f} | R {ep_ret:.2f}")

        if epoch % EVAL_INTERVAL == 0:
            with open("models/ssm_actor_critic_1d_g95" + str(epoch) + ".pkl", "wb") as f:
                pickle.dump(agent.state_dict(), f)
            print("--- Evaluation ---")
            play_once(agent)

    # final render
    print("=== Final Policy Evaluation ===")
    play_once(agent)

    # ---------------------- Plot metrics -------------------------- #
    smooth_w = 25
    def smooth(x):
        if len(x) < smooth_w:
            return np.array(x)
        filt = np.ones(smooth_w) / smooth_w
        return np.convolve(x, filt, mode='valid')

    x_axis = np.arange(len(smooth(critic_hist)))
    plt.figure(figsize=(8,4))
    plt.plot(x_axis, smooth(critic_hist), label='Critic Loss')
    plt.plot(x_axis, smooth(actor_hist),  label='Actor Loss')
    plt.plot(x_axis, smooth(return_hist), label='Mean Return')
    plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout()
    plt.title('Training Curves (window=25)')
    plt.show()

if __name__ == "__main__":
    device = DEVICE
    train()
