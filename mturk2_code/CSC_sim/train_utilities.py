import os, json, shutil, pickle, random, numpy as np, torch, torch.nn.functional as F, matplotlib
from dataclasses import dataclass, asdict
from typing import Optional, Sequence
from torch.distributions import Categorical
from einops import rearrange, repeat
from ColorShapeSpace_sim import CSC2Env, make_csc2_vector
from combined_model import Deform, QAgent, AgentConfig
matplotlib.use('TkAgg')

# =============================== Config =====================================

@dataclass
class TrainConfig(AgentConfig):
    # training/runtime
    batch_size: int = 10
    epochs: int = 3000
    eval_interval: int = 500
    lr: float = 1e-2
    gamma: float = 0.0              # TD discount
    dims: int = 1                   # color-shape space: 1D or 2D
    max_trials: int = 64 * 5        # default aligns with chunk_size*5
    num_choices: int = 1
    seed_override: Optional[int] = None
    alpha_entropy: float = 0.01     # kept for completeness
    self_action: bool = False
    temperature: float = 1.0
    # saving
    save_root: str = "models"
    run_name: str = "default"

# ========================== Feature & Cache utils ============================

def build_value_obs(obs_a: np.ndarray, cfg: TrainConfig) -> torch.Tensor:
    """angles (B,k,2) -> sin|cos features  ==> (B*k, 2*cfg.dims )
       (keeps legacy shaping semantics)."""
    B, k, _ = obs_a.shape
    x = torch.from_numpy(obs_a).to(cfg.device) if not torch.is_tensor(obs_a) else obs_a.to(cfg.device)
    feat = torch.stack([torch.sin(x), torch.cos(x)], dim=-1).flatten(-2) # make interleaved
    feat = feat.view(B * k, cfg.dims * 2)                                   # legacy: drops extra dims if dims==1
    return feat

def collapse_agent_cache(cache, B, action):
    s = rearrange(cache.conv_state, "(b k) x y->b k x y", b=B); cache.conv_state = s[torch.arange(B), action]
    s = rearrange(cache.ssm_state,  "(b k) x y z->b k x y z", b=B); cache.ssm_state = s[torch.arange(B), action]
    return cache

def expand_agent_cache(cache, B, k):
    if cache is not None:
        cache.conv_state = repeat(cache.conv_state, 'b x y -> (b k) x y', k=k).contiguous()
        cache.ssm_state  = repeat(cache.ssm_state,  'b x y z -> (b k) x y z', k=k).contiguous()
    return cache

# ========================== Shared policy forward ============================

def apply_deform_inplace(flat_inp: torch.Tensor, deformer: Deform, cfg: TrainConfig, add_noise: bool):
    """
    Returns a new tensor (same shape) with deformed perceptual dimensions replaced.
    """
    p = deformer(flat_inp[:, :cfg.percept_dim])
    if add_noise:
        p = p + torch.normal(mean=0., std=0.1, size=p.shape, device=p.device)
    return torch.cat([p, flat_inp[:, cfg.percept_dim:]], dim=-1)

def _forward_and_sample(agent: QAgent, flat_inp: torch.Tensor, prev_r, B: int, k: int):
    agent.cache = expand_agent_cache(agent.cache, B, k)
    prev_r = torch.tile(prev_r.reshape(B, 1), (1, k)).flatten() #Bk
    _, logits = agent(flat_inp.unsqueeze(0), prev_r.unsqueeze(0), k=k)     # (1,B,k)
    logits = logits.squeeze(0)
    act = Categorical(logits=logits).sample()
    return logits, act

# ============================== Rollout (Q) ==================================

@torch.no_grad()
def collect_batch_q(vec, agent: QAgent, deformer: Deform, cfg: TrainConfig, dev="cpu"):
    """One full episode (T=cfg.max_trials) across vector env."""
    B, k, T = vec.num_envs, cfg.num_choices, cfg.max_trials
    obs_a, _ = vec.reset()
    pd = cfg.device
    cfg.device = dev
    agent = agent.to(dev)
    deformer = deformer.to(dev)
    prev_r = torch.zeros(B, device=cfg.device)
    actions = torch.empty(T, B, dtype=torch.long, device=cfg.device)
    rewards = torch.zeros(T + 1, B, device=cfg.device)
    obs_seq = torch.empty(T, B, 2 * cfg.dims, device=cfg.device)
    agent.sequential = True
    for t in range(T):
        obs = build_value_obs(obs_a, cfg)  # Bk, dims * 2
        flat = apply_deform_inplace(obs, deformer, cfg, add_noise=True)
        logits, act = _forward_and_sample(agent, flat, prev_r, B, k)
        actions[t] = act
        obs_seq[t] = obs.reshape((B, k, cfg.dims * 2))[torch.arange(B), act, :]
        obs_a, r, _, _, _ = vec.step(act.cpu().numpy())
        agent.cache = collapse_agent_cache(agent.cache, k, act)
        rewards[t + 1] = torch.from_numpy(r).to(cfg.device); prev_r = rewards[t + 1]
    cfg.device = pd
    agent = agent.to(pd)
    deformer = deformer.to(pd)
    agent.reset()
    agent.sequential = False
    return obs_seq.to(cfg.device), rewards.float().to(cfg.device), 0.0

# ============================== Evaluation ===================================

def play_once(agent: QAgent, deformer: Deform, cfg: TrainConfig, *, trials: int, disp_steps: int = 30,
              render=True, seed=None, k=None):
    pd = cfg.device
    cfg.device = "cpu"
    agent = agent.to("cpu")
    deformer = deformer.to("cpu")
    if cfg.seed_override is not None: seed = cfg.seed_override
    if k is None: k = agent.n_actions
    n = 72 if cfg.dims == 1 else 36
    env = CSC2Env(n=n, trials=trials, render=render, render_reward=render, k=k, seed=seed, one_d=(cfg.dims == 1))
    obs_a, _ = env.reset()
    prev_r = torch.zeros(1, device=cfg.device)
    agent.reset(); agent.sequential = True
    total_r, step, states, ret_hist = 0.0, 0, [], []
    while True:
        if (step + 1) % disp_steps == 0: env.render_reward = env.render_flag = render
        flat = build_value_obs(obs_a[np.newaxis, ...], cfg)
        flat = apply_deform_inplace(flat, deformer, cfg, add_noise=False)
        with torch.no_grad():
            _, action = _forward_and_sample(agent, flat, prev_r, 1, k)
        states.append(agent.hidden[action])
        obs_a, r, done, _, _ = env.step(int(action))
        agent.cache = collapse_agent_cache(agent.cache, 1, action)
        prev_r = torch.tensor(r, device=cfg.device).unsqueeze(0)
        total_r += r; step += 1; ret_hist.append(r)
        env.render_reward = env.render_flag = False
        if done: break
    L100 = np.stack(ret_hist[-100:]).mean() if len(ret_hist) >= 100 else np.mean(ret_hist)
    env.close()
    print(f"Rendered episode total return: {total_r}\nLast 100 Average R: {L100:.3f}")
    cfg.device = pd
    agent = agent.to(pd)
    deformer = deformer.to(pd)
    return torch.concatenate(states, dim=0).detach().cpu()

# ============================== Saving utils =================================
def save_checkpoint(cfg: TrainConfig, agent: QAgent, deform: Deform, epoch: int, tag: str):
    """Write {save_root}/{run_name}_{tag}_e{epoch}/ with config.json + state_dicts."""
    d = os.path.join(cfg.save_root, f"{cfg.run_name}_{tag}_e{epoch:04d}")
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(agent.state_dict(), os.path.join(d, "agent.pt"))
    torch.save(deform.state_dict(), os.path.join(d, "deform.pt"))

