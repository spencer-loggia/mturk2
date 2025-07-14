import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.vector import SyncVectorEnv
from xarray_einstats.einops import rearrange

from ColorShapeSpace_sim import CSC2Env

from loss import ActorCritic
from SSM import SSD_cell, ssd
from neurotools.util import is_converged
import numpy as np
from matplotlib import pyplot as plt

# --------------------------- Hyper‑parameters ---------------------------- #
BATCH_SIZE     = 4
EPOCHS         = 500
EVAL_INTERVAL  = 500      # render every N epochs
CHUNK_SIZE     = 32
GAMMA          = 0.99
MAX_TRIALS     = 100
EVAL_TRIALS    = 25       # trials when rendering policy
ALPHA_ENTROPY  = 0.01
LR             = 3e-4
N_UNITS        = 8
HEAD_DIM       = 4
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------- Agent & ValueNet --------------------------- #
class ActorAgent(nn.Module):
    """Little agents for testing. Won't perform in this partially observable space"""
    def __init__(self, obs_dim, n_actions, n_units, n_features, state_size, p_heads, train=True):
        super().__init__()
        self.input = nn.Linear(obs_dim, n_units * n_features)
        self.model = SSD_cell(n_units, n_features, p_heads, state_size, device=DEVICE)
        self.output = nn.Linear(n_units, n_actions)
        self.train = train
        self.hidden = []

    def forward(self, obs):          # obs: (t, B,k,2)
        t, B, k, _ = obs.shape
        obs = obs.reshape(t * B, -1)
        x = torch.relu(self.input(obs))
        x = rearrange(x, '(tB)(kf) -> tBkf', k=k, B=B)
        y = self.model.forward(x) # <t, b, s>
        self.hidden = y.detach().clone()
        y = y.reshape(t * B, self.n_units)
        y = torch.relu(y)
        action_logit = self.output(y) # <tb, a>
        return action_logit.reshape(t, B, self.n_actions)

    def reset(self):
        self.hidden = []
        self.model.reset()


class ValueNet(nn.Module):
    """Little critic for testing. Won't perform in this partially observable space:
        same flattened observation -> scalar value."""
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        :param obs: tensor (time, batch, obs_dim + hidden)
        """
        t =  obs.shape[0]
        b = obs.shape[1]
        v = self.net(obs.reshape(t *b , -1))
        return v.reshape(t, b)

# --------------------------- Vector helpers ----------------------------- #

def make_vec(batch):
    """Create a vectorised environment batch for training."""
    return SyncVectorEnv([lambda: CSC2Env(trials=MAX_TRIALS, render=False, batched=True) for _ in range(batch)])

def collect_batch(vec, agent, value_net, chunk_size, device):
    """RUsing vectorized environment in batched mode - in parallel genrate all states and actions from all time in
    multiple environments"""
    B = vec.num_envs
    T_max = MAX_TRIALS
    k = agent.n_actions

    assert T_max % chunk_size == 0 # number of trials must be a mulitple of chunk size

    obs, _ = vec.reset()

    # get forward actions
    obs_batch, _ = vec.reset()  # batch, time, k, 2
    obs_batch = torch.from_numpy(obs_batch).float().to(device)
    obs_batch = obs_batch.transpose(0, 1) # time, batch, k, 2
    act_logits = agent.forward(obs_batch) # time, batch, k
    act_logits = act_logits.reshape(T_max * B, k)

    # compute distribution
    dist = Categorical(logits=act_logits)
    actions = dist.sample() # tb
    logp = dist.log_prob(actions)
    entr = dist.entropy()

    # compute critic value estimates and true reward
    obs_batch = obs_batch.reshape(B, T_max, -1)
    vals = value_net.forward(torch.concatenate([obs_batch, agent.hidden], dim=2))
    reward = torch.from_numpy(vec.step(actions))
    # no mask necessary since we know we have alive trials divisible by chunks

    # reshape to time and batch
    reward = reward.reshape(T_max, B)
    vals = vals.reshape(T_max, B)
    logp = logp.reshape(T_max, B)
    entr = entr.reshape(T_max, B)

    # episode returns per env (before padding) for monitoring
    ep_returns = reward.sum(dim=0).mean().item()

    return (reward.flatten(), vals.flatten(), logp.flatten(), entr.flatten(), None, ep_returns)

# --------------------------- Play & Render ------------------------------ #

def play_once(agent):
    env = CSC2Env(trials=EVAL_TRIALS, render=True, render_reward=True)
    obs, _ = env.reset()
    total_r = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = agent(obs_t)
        action = torch.argmax(logits, dim=-1).item()
        obs, r, done, _, _ = env.step(action)
        total_r += r
        if done:
            break
    env.close()
    print(f"Rendered episode return: {total_r}")

# --------------------------- Training ---------------------------------- #

def train(agent=None, loss_function=None):
    k = CSC2Env().k
    obs_dim = 2 * k

    agent     = ActorAgent(obs_dim, k, N_UNITS, 4, 8, 4).to(DEVICE)
    value_net = ValueNet(obs_dim + N_UNITS).to(DEVICE)
    loss_fn   = ActorCritic(gamma=GAMMA, alpha=ALPHA_ENTROPY)

    act_optim = torch.optim.Adam(agent.parameters(), lr=LR)
    crit_optim =  torch.optim.Adam(value_net.parameters())
    vec = make_vec(BATCH_SIZE)

    critic_hist, actor_hist, return_hist = [], [], []

    for epoch in range(1, EPOCHS + 1):
        (rews, vals, logp, entr, mask, ep_ret) = collect_batch(
            vec, agent, value_net, CHUNK_SIZE, DEVICE)

        critic_loss, actor_loss = loss_fn(rews, vals, logp, entr, mask)
        loss = critic_loss + actor_loss
        act_optim.zero_grad(); crit_optim.zero_grad(); loss.backward(); act_optim.step(); crit_optim.step()

        # dynamically adjust LR
        act_optim, adone = is_converged(actor_hist, act_optim, 1, epoch, max_lr=.1)
        crit_optim, cdone = is_converged(critic_hist, crit_optim, 1, epoch, max_lr=.1)

        critic_hist.append(critic_loss.item())
        actor_hist.append(actor_loss.item())
        return_hist.append(ep_ret)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Critic {critic_loss.item():.3f} | Actor {actor_loss.item():.3f} | R {ep_ret:.2f}")

        if epoch % EVAL_INTERVAL == 0:
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
    plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout();
    plt.title('Training Curves (window=25)')
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()