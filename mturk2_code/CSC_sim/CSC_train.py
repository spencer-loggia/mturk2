import pickle

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.vector import SyncVectorEnv
from einops import rearrange

from ColorShapeSpace_sim import CSC2Env, make_csc2_vector

from loss import ActorCritic
from SSM import SSM, SSMConfig, InferenceCache
from neurotools.util import is_converged
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# --------------------------- Hyper‑parameters ---------------------------- #
BATCH_SIZE     = 10      # number of parallel environments
EPOCHS         = 5000
EVAL_INTERVAL  = 5000      # render every N epochs
CHUNK_SIZE     = 32
GAMMA          = 0.95
MAX_TRIALS     = CHUNK_SIZE * 30
EVAL_TRIALS    = 25       # trials when rendering policy
ALPHA_ENTROPY  = 0.1
LR             = .01
N_UNITS        = 16
HEAD_DIM       = 8
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'


ssm_configuration =SSMConfig(
        d_model=N_UNITS, d_state=8, d_conv=2, expand=1, headdim=2, chunk_size=64,
    )

# --------------------------- Agent & ValueNet --------------------------- #
class ActorAgent(nn.Module):
    """Little agents for testing. Won't perform in this partially observable space"""
    def __init__(self, obs_dim, n_actions, n_units, n_features, state_size, p_heads, train=True):
        super().__init__()
        self.input = nn.Linear(obs_dim, n_units, device=DEVICE)
        self.model = SSM(ssm_configuration, device=DEVICE)
        self.output = nn.Linear(n_units, n_actions, device=DEVICE)
        self.device = device
        self.hidden = []
        self.n_actions = n_actions
        self.n_units = n_units
        self.cache = None

    def _action_from_hidden(self, y):
        t, B, _ = y.shape
        self.hidden = y.detach().clone()
        y = y.reshape(t * B, self.n_units)
        y = torch.relu(y)
        action_logit = self.output(y)  # <tb, a>
        return action_logit.reshape(t, B, self.n_actions)

    def forward(self, obs, save_params=False):   # obs: (t, B,n_choices*4)
        t, B, _ = obs.shape
        if self.train and self.cache is not None:
            self.cache = InferenceCache.alloc(batch_size=B, args=ssm_configuration)
        k = self.n_units
        obs = obs.reshape(t * B, -1)
        x = torch.relu(self.input(obs))
        x = rearrange(x,'(t B) (k f) -> t B k f', t=t, B=B, k=k)
        y, h = self.model.forward(x)  # <t, b, s>
        self.cache = h
        return self._action_from_hidden(y)

    def recompute(self):
        y = self.model.recompute()
        return self._action_from_hidden(y)

    def reset(self):
        self.cache = None
        self.hidden = []


class ValueNet(nn.Module):
    """Little critic for testing. Won't perform in this partially observable space:
        same flattened observation -> scalar value."""
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden, device=DEVICE), nn.ReLU(),
            nn.Linear(hidden, hidden, device=DEVICE), nn.ReLU(),
            nn.Linear(hidden, 1, device=DEVICE)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        :param obs: tensor (time, batch, obs_dim + hidden)
        """
        t =  obs.shape[0]
        b = obs.shape[1]
        v = self.net(obs.reshape(t * b , -1))
        return v.reshape(t, b)


# --------------------------- Vector helpers ----------------------------- #

def make_vec(batch):
    """Create a vectorised environment batch for training."""
    return SyncVectorEnv([lambda: CSC2Env(trials=MAX_TRIALS, render=False, batched=False) for _ in range(batch)])


def collect_batch(vec, agent, value_net, chunk_size, device):
    """RUsing vectorized environment in batched mode - in parallel genrate all states and actions from all time in
    multiple environments"""
    B = vec.num_envs
    T_max = MAX_TRIALS
    k = agent.n_actions

    assert T_max % chunk_size == 0 # number of trials must be a mulitple of chunk size

    obs, _ = vec.reset()

    # get observations in parallel
    obs_a, _ = vec.reset()  # batch, 1, k, 2
    obs_a = obs_a.reshape(B, k, 2)
    obs_a = torch.from_numpy(obs_a).float().to(device)
    obs = torch.concatenate([torch.sin(obs_a), torch.cos(obs_a)], dim=-1) # b k 4
    obs = obs.reshape(1, B, k * 4)
    actions = torch.empty(T_max, B, dtype=torch.int, device=DEVICE)
    rewards = torch.zeros(T_max + 1, B, device=DEVICE)  # will be added to next input
    obs_batch = torch.zeros(T_max, B, k * 4, dtype=torch.float32, device=DEVICE)
    # sequential pass to get all rewards
    with torch.no_grad():
        for i in range(MAX_TRIALS):
            obs_batch[i] = obs[0]
            act_logits = agent.forward(torch.concatenate([obs, rewards[i][None, :, None]], dim=2),
                                       save_params=True) # time, batch, k
            act_logits = act_logits.reshape(1 * B, k)
            # compute distribution
            dist = Categorical(logits=act_logits)
            actions[i] = dist.sample() # need to recall actions so we can evaluate them with the ssd path
            obs_a, reward, _, _, _ = vec.step(actions[i].T.detach().cpu().numpy())
            rewards[i + 1] = torch.from_numpy(reward).float().to(DEVICE)

    # compute ssd
    agent.reset()
    act_logits = agent.forward(torch.concatenate([obs_batch, rewards[:-1].unsqueeze(2)], dim=2))
    act_logits = act_logits.reshape(T_max * B, -1)
    dist = Categorical(logits=act_logits)
    logp = dist.log_prob(actions.flatten())
    entr = dist.entropy()

    # compute critic value estimates and true reward
    obs_batch = obs_batch.reshape(B, T_max, -1).transpose(0,1) # time, batch, obs
    vals = value_net.forward(torch.concatenate([obs_batch, agent.hidden], dim=2))
    reward = rewards[1:] # eliminate the initial blank reward for loss compute.
    # no mask necessary since we know we have alive trials divisible by chunks

    # reshape to time and batch
    logp = logp.reshape(T_max, B)
    entr = entr.reshape(T_max, B)

    # episode returns per env (before padding) for monitoring
    ep_returns = reward.mean().item()
    return (reward, vals, logp, entr, None, ep_returns)

# --------------------------- Play & Render ------------------------------ #

def play_once(agent):
    env = CSC2Env(trials=EVAL_TRIALS, render=True, render_reward=True)
    obs, _ = env.reset()
    total_r = 0
    last_r = torch.zeros((1, 1), device=DEVICE)
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        obs_t = torch.concatenate([torch.sin(obs_t), torch.cos(obs_t)], dim=-1)
        obs_t = obs_t.reshape(1, 1, -1)
        obs_t = torch.concatenate([obs_t, last_r.unsqueeze(-1)], dim=-1)
        with torch.no_grad():
            logits = agent(obs_t)
        action = torch.argmax(logits, dim=-1).item()
        obs, r, done, _, _ = env.step(action)
        last_r = torch.tensor(r).float().to(device=DEVICE).reshape((1, 1))
        total_r += r
        if done:
            break
    env.close()
    print(f"Rendered episode return: {total_r}")

# --------------------------- Training ---------------------------------- #

def train(agent=None, loss_function=None):
    k = CSC2Env().k
    obs_dim = 4 * k

    agent     = ActorAgent(obs_dim + 1, k, N_UNITS, 6, 64, 32).to(DEVICE)
    value_net = ValueNet(obs_dim + N_UNITS).to(DEVICE)
    loss_fn   = ActorCritic(gamma=GAMMA, alpha=ALPHA_ENTROPY)

    act_optim = torch.optim.Adam(agent.parameters(), lr=LR)
    crit_optim =  torch.optim.Adam(value_net.parameters())
    vec = make_csc2_vector(BATCH_SIZE, batched=False, trials=MAX_TRIALS) # use custom environment that allows reward vectors

    critic_hist, actor_hist, return_hist = [], [], []
    for epoch in range(1, EPOCHS + 1):
        (rews, vals, logp, entr, mask, ep_ret) = collect_batch(
            vec, agent, value_net, CHUNK_SIZE, DEVICE)

        critic_loss, actor_loss = loss_fn(rews, vals, logp, entr, mask)
        loss = critic_loss + actor_loss
        act_optim.zero_grad(); crit_optim.zero_grad(); loss.backward(); act_optim.step(); crit_optim.step()
        agent.reset()

        # dynamically adjust LR
        act_optim, adone = is_converged(actor_hist, act_optim, BATCH_SIZE, epoch, max_lr=.1)
        crit_optim, cdone = is_converged(critic_hist, crit_optim, BATCH_SIZE, epoch, max_lr=.1)

        critic_hist.append(critic_loss.detach().cpu().item())
        actor_hist.append(actor_loss.detach().cpu().item())
        return_hist.append(ep_ret)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | Critic {critic_loss.item():.3f} | Actor {actor_loss.item():.3f} | R {ep_ret:.2f}")

        if epoch % EVAL_INTERVAL == 0:
            with open("/home/bizon/Projects/mturk2/mturk2_code/CSC_sim/models/ssm_csc.pkl", "wb") as f:
                pickle.dump(agent, f)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()