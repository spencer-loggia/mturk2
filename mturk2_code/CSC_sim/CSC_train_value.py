import pickle
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
from einops import rearrange, repeat
from gymnasium.vector import SyncVectorEnv
from neurotools.util import is_converged

from ColorShapeSpace_sim import CSC2Env, make_csc2_vector
from loss import TemporalDifference                    # TD(λ=0) loss impl
from CogSSM import SSM, SSMConfig, InferenceCache
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

# ----------------------------- Hyper‑parameters -----------------------------
BATCH_SIZE     = 10         # parallel environments
EPOCHS         = 2000
EVAL_INTERVAL  = 500         # render every N epochs
CHUNK_SIZE     = 64          # SSM chunk size
DIMS           = 1
MAX_TRIALS     = CHUNK_SIZE * 5
GAMMA          = 0.0         # discount for TD target
ALPHA_ENTROPY  = 0.01        # entropy bonus coefficient
LR             = 1e-2
N_UNITS        = 2
HEAD_DIM       = 2
NUM_CHOICES    = 1
DEVICE         = "cpu" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED_OVERRIDE = None
SELF_ACTION = False

# ----------------------------- SSM config -----------------------------------
ssm_config = SSMConfig(
    d_model=N_UNITS,
    d_state=1,
    d_conv=2,
    expand=1,
    headdim=HEAD_DIM,
    chunk_size=CHUNK_SIZE,
)

# ---------------------------------------------------------------------------
# Helper : feature construction (angle ➜ sin|cos, +previous reward)
# ---------------------------------------------------------------------------

def build_value_obs(obs_a: np.ndarray, prev_r: torch.Tensor) -> torch.Tensor:
    """Convert polar‐angle features + previous reward into (B*k, 5) tensor.

    obs_a : ndarray (B, k, 2)  – polar coords for each option
    prev_r: tensor   (B,)      – reward obtained on previous trial
    returns: tensor  (B*k, 5)  – sin, cos, reward features per option
    """
    B, k, _ = obs_a.shape
    feat = torch.from_numpy(obs_a).float().to(DEVICE)           # B,k,2
    feat = torch.cat([torch.sin(feat), torch.cos(feat)], dim=-1)  # B,k,4
    feat = feat.view(B * k, DIMS * 2)
    r = prev_r.repeat_interleave(k).unsqueeze(1)                 # B*k,1
    return torch.cat([feat, r], dim=1)                           # B*k,5

# ---------------------------------------------------------------------------
# Model: Q‑value network with internal SSM state + temperature scaling
# ---------------------------------------------------------------------------

class QAgent(nn.Module):
    """Maps (sin,cos,reward) features -> scalar Q‑value for each option.

    The policy is derived via soft‑max(Q / τ) with learnable temperature τ.
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim   = obs_dim

        self.input = nn.Linear(obs_dim, N_UNITS, bias=True, device=DEVICE)
        self.ssm   = SSM(ssm_config, device=DEVICE)
        self.head  = torch.nn.Parameter(torch.tensor([0.], device=DEVICE)) # nn.Linear(N_UNITS, 1, bias=False, device=DEVICE)    # scalar Q
        # temperature parameter (log parameterization for positivity)
        self.log_tau = nn.Parameter(torch.zeros((1), device=DEVICE))
        self.sa = torch.tensor([0.], device=DEVICE)
        self.sa_net = torch.nn.Linear(in_features=self.ssm.args.d_inner * self.ssm.args.d_state, out_features=DIMS)
        self.hidden = None  # b, p, h, s

        # Inference cache for sequential rollout
        self.sequential = False
        self.cache      = None

    # ---------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, *, k: int):
        """obs: (T, B*k, obs_dim)   → values (T,B,k) and logits (T,B,k)"""
        T, Bk, _ = obs.shape

        # if operating with self action need to add action to network.
        if self.sequential and self.cache is None:
            self.cache = InferenceCache.alloc(batch_size=Bk, args=ssm_config, device=DEVICE)
        if self.sequential:
            h = self.cache
        else:
            h = None
        x = (self.input(obs.float().view(T * Bk, -1)))          # (T*Bk,N_UNITS)
        x = rearrange(x, '(t b) s -> t b s', t=T, b=Bk)           # (T,Bk,N_UNITS)

        y, self.cache = self.ssm(x.transpose(0, 1), h)   # (T,Bk,N_UNITS)
        self.hidden = self.cache.ssm_state.clone()
        y = y.transpose(0, 1)

        q = torch.sum(y, dim=-1) * torch.nn.functional.softplus(self.head)                     # (T,Bk)
        q = q.view(T, -1, k)                                      # (T,B,k)

        tau = torch.exp(self.log_tau)
        logits = q / tau
        return q, logits

    # ---------------------------------------------------------------------
    def reset(self):
        self.sequential = False
        self.cache = None


def _collapse_agent_cache(cache, B, action):
    """
    Collapse cached states to value that was actually measured
    goes from have leading dims of batch * k -> batch
    """
    s = cache.conv_state
    s = rearrange(s, "(b k) x y->b k x y", b=B)
    cache.conv_state = s[torch.arange(B), action]
    s = cache.ssm_state
    s = rearrange(s, "(b k) x y z -> b k x y z", b=B)
    cache.ssm_state = s[torch.arange(B), action]
    return cache


def _expand_agent_cache(cache, B, k):
    """
    goes from have leading dims of batch * k -> batch
    """
    if cache is not None:
        s = cache.conv_state
        s = repeat(s, 'b x y -> (b k) x y', k=k)
        cache.conv_state = s.contiguous()
        s = cache.ssm_state
        s = repeat(s, 'b x y z -> (b k) x y z', k=k)
        cache.ssm_state = s.contiguous()
    return cache

# ---------------------------------------------------------------------------
# Roll‑out to collect trajectories for Q‑learning
# ---------------------------------------------------------------------------

def collect_batch_q(vec, agent: QAgent):
    """Generate one full episode (T=MAX_TRIALS) of experience.

    Returns
    -------
    q_pred   : (T,B)  – predicted Q for the *chosen* action
    rewards  : (T,B)  – actual rewards (t aligned with q_pred)
    entropy  : scalar – mean policy entropy (for bonus)
    """
    B   = vec.num_envs
    k   = agent.n_actions
    T   = MAX_TRIALS

    obs_a, _ = vec.reset()                            # (B,k,2)
    prev_r   = torch.zeros(B, device=DEVICE)

    # storage
    actions = torch.empty(T, B,   dtype=torch.long,  device=DEVICE)
    rewards = torch.zeros(T + 1, B, device=DEVICE)    # include t=0 dummy
    obs_seq = torch.empty(T,  B, 2 * DIMS + 1, device=DEVICE)

    agent.sequential = True

    # --------------- rollout ---------------
    with torch.no_grad():
        for t in range(T):
            inp = build_value_obs(obs_a, prev_r)           # (B*k,5)
            agent.cache = _expand_agent_cache(agent.cache, B, k)
            q, logits = agent(inp.unsqueeze(0), k=k)       # (1,B,k)
            logits = logits.squeeze(0)                     # (B,k)

            dist = Categorical(logits=logits)
            act = dist.sample()                            # (B,)
            actions[t] = act
            # remember actually selected action
            obs_seq[t] = inp.reshape(B, k, 2 * DIMS + 1)[np.arange(B), act, :]

            obs_a, r, _, _, _ = vec.step(act.cpu().numpy())
            # by default the agent save a cache with BATCH indexes, but we want to collapse to the index of the
            # item we actually chose:
            agent.cache = _collapse_agent_cache(agent.cache, B, act)

            rewards[t + 1] = torch.from_numpy(r).to(DEVICE)  # align +1
            prev_r = rewards[t + 1]
    # --------------- prepare for backward pass ---------------
    agent.reset(); agent.sequential = False
    return obs_seq, rewards[1:].float(), 0.

# ---------------------------------------------------------------------------
# Evaluation episode (greedy policy, rendered)
# ---------------------------------------------------------------------------

def play_once(agent: QAgent, *, trials: int, disp_steps: int = 30, render=True, seed=None, k=None):
    if SEED_OVERRIDE is not None:
        seed = SEED_OVERRIDE
    if k is None:
        k = agent.n_actions
    if DIMS==1:
        n = 72
    else:
        n = 36
    env = CSC2Env(n=n, trials=trials, render=render, render_reward=render, k=k, seed=seed, one_d=DIMS==1)
    obs_a, _ = env.reset()                             # (k,2) because B=1
    prev_r = torch.zeros(1, device=DEVICE)

    agent.reset(); agent.sequential = True
    total_r = 0; step = 0

    states = []
    ret_hist = []
    while True:
        if (step + 1) % disp_steps == 0:
            env.render_reward = render; env.render_flag = render

        inp = build_value_obs(obs_a[np.newaxis, ...], prev_r).unsqueeze(0)  # (1,k,5)
        agent.cache = _expand_agent_cache(agent.cache, 1, env.k)
        with torch.no_grad():
            _, logits = agent(inp, k=env.k)                                 # (1,1,k)
        logits = logits.squeeze(0)                     # (B,k)
        dist = Categorical(logits=logits)
        action = dist.sample()  # (B,)
        states.append(agent.hidden[action])

        obs_a, r, done, _, _ = env.step(action)

        # by default the agent save a cache with BATCH indexes, but we want to collapse to the index of the
        # item we actually chose:
        agent.cache = _collapse_agent_cache(agent.cache, 1, action)

        prev_r = torch.tensor(r, device=DEVICE).unsqueeze(0)
        total_r += r; step += 1
        ret_hist.append(r)
        env.render_reward = False; env.render_flag = False
        if done:
            break

    L100 = np.stack(ret_hist[-100:]).mean()
    env.close()
    print(f"Rendered episode total return: {total_r}")
    print(f"Last 100 Average R: {L100}")
    return torch.concatenate(states, dim=0).detach().cpu()

# ---------------------------------------------------------------------------
# Training loop – TD(0) + entropy bonus
# ---------------------------------------------------------------------------

def train():
    env_proto = CSC2Env(k=NUM_CHOICES)
    k = env_proto.k

    obs_dim = 2 * DIMS + 1                            # per‑option features
    # if SELF_ACTION:
    #     obs_dim = obs_dim + DIMS

    agent = QAgent(obs_dim, k).to(DEVICE)

    td_loss_fn = TemporalDifference(gamma=GAMMA, normalize=True)
    optimizer  = torch.optim.Adam(agent.parameters(), lr=LR)

    if DIMS == 1:
        one_d = True
    elif DIMS == 2:
        one_d = False
    else:
        raise ValueError

    loss_hist, ret_hist = [], []

    storage = dict()

    for epoch in range(1, EPOCHS + 1):
        agent.reset()
        if DIMS == 1:
            n = 72
        else:
            n = 36
        vec = make_csc2_vector(BATCH_SIZE, n=n, batched=False, trials=MAX_TRIALS, k=NUM_CHOICES, one_d=one_d,
                               seeds=[SEED_OVERRIDE] * BATCH_SIZE)

        obs_seq, rewards, _ = collect_batch_q(vec, agent)

        # add replay
        keys = random.sample(list(storage), min(len(storage), 5 * BATCH_SIZE))
        r_obs = [obs_seq]
        r_rewards = [rewards]
        for k in keys:
            r_obs.append(storage[k][0])
            r_rewards.append(storage[k][1])
        r_obs = torch.concatenate(r_obs, dim=1)
        r_rewards = torch.concatenate(r_rewards, dim=1)

        # process forward with SSM
        q_all, logits_all = agent(r_obs, k=1)  # (T,B,k)
        # gather Q for realised actions
        q_pred = q_all.squeeze(-1)  # (T,B)
        q_pred = q_pred / (torch.arange(MAX_TRIALS, device=DEVICE, dtype=torch.float).unsqueeze(1) + 1)

        td_loss = td_loss_fn(q_pred, r_rewards)
        loss = td_loss # - ALPHA_ENTROPY * entropy

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        optimizer, adone = is_converged(loss_hist, optimizer, BATCH_SIZE, epoch, max_lr=.01)

        # prune storage to size
        keys = random.sample(list(storage), max(0, len(storage) - 20 * BATCH_SIZE))
        # pop and collect
        storage = {k: storage.pop(k) for k in keys}
        # add new storage
        for b in range(BATCH_SIZE):
            storage[str(epoch) + str(b)] = (obs_seq[:, b:(b+1), ...].detach(), rewards[:, b:(b+1)].detach())

        loss_hist.append(td_loss.item())
        ret_hist.append(rewards[-100:].mean().item())

        if epoch % 10 == 0:
            print(f"Ep {epoch:04d} | TD {td_loss.item():.3f} | R {ret_hist[-1]:.2f}")

        if epoch % EVAL_INTERVAL == 0:
            with open("models/p1h2s1_ssm_csc_q_1d" + str(epoch) + ".pkl", "wb") as f:
                pickle.dump(agent.state_dict(), f)
            print("--- Evaluation ---")
            play_once(agent, trials=MAX_TRIALS, render=False, k=max(agent.n_actions, 2))

    # final render ------------------------------------------------------
    print("=== Final Policy Evaluation ===")
    play_once(agent, trials=MAX_TRIALS, render=False)

    # ---------------------- Plot metrics ------------------------------
    if len(loss_hist) >= 25:
        smooth_w = 25
        filt = np.ones(smooth_w) / smooth_w
        loss_sm = np.convolve(loss_hist, filt, mode='valid')
        ret_sm  = np.convolve(ret_hist,  filt, mode='valid')

        plt.figure(figsize=(8,4))
        plt.plot(loss_sm, label='TD Loss')
        plt.plot(ret_sm,  label='Mean Return')
        plt.legend(); plt.xlabel('Epoch (smoothed)'); plt.tight_layout()
        plt.title('Training Curves (window=25)')
        plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train()