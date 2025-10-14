# behavior_env_2d_csv.py
import re
import math
from functools import partial
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

TAU = 2 * math.pi


# ------------------------------- CSV → Episode -------------------------------

def _detect_option_columns(df: pd.DataFrame) -> tuple[list[str], list[str], int]:
    """Return sorted s_cols, c_cols, and k (number of options)."""
    s_cols = sorted([c for c in df.columns if re.fullmatch(r"s\d+", c)], key=lambda x: int(x[1:]))
    c_cols = sorted([c for c in df.columns if re.fullmatch(r"c\d+", c)], key=lambda x: int(x[1:]))
    if not s_cols or not c_cols:
        raise ValueError("CSV must contain s1..sK and c1..cK columns.")
    k = min(len(s_cols), len(c_cols))
    return s_cols[:k], c_cols[:k], k


def load_episode_from_csv_2d_file(path: str, max_trials: int):
    """
    Parse a single CSV file with columns:
      c1,s1,c2,s2,...,cK,sK, choice, reward

    Returns:
      dict with:
        'angles'  : (T, k, 2) float32  (theta, phi) with theta=2π*s/max(s), phi=2π*c/max(c)
        'choices' : (T,) int64
        'rewards' : (T,) float32
    """
    df = pd.read_csv(path, sep="\t")
    if "choice_idx" not in df.columns or "reward" not in df.columns:
        raise ValueError("CSV must contain 'choice_idx' and 'reward' columns.")
    s_cols, c_cols, k = _detect_option_columns(df)

    s_idx = df[s_cols].to_numpy(dtype=np.int64)  # (T,k)
    c_idx = df[c_cols].to_numpy(dtype=np.int64)  # (T,k)

    # denominators as requested: max across the file (per index family)
    s_den = max(1, int(s_idx.max()))
    c_den = max(1, int(c_idx.max()))

    theta = (TAU * s_idx.astype(np.float32)) / float(s_den)  # (T,k)
    phi   = (TAU * c_idx.astype(np.float32)) / float(c_den)  # (T,k)
    angles = np.stack([theta, phi], axis=-1).astype(np.float32)  # (T,k,2)

    choices = df["choice_idx"].to_numpy(dtype=np.int64)  # (T,)
    rewards = df["reward"].to_numpy(dtype=np.float32)  # (T,)

    if (choices < 0).any() or (choices >= k).any():
        raise ValueError(f"'choice' values must be in [0, {k}).")

    return {"angles": angles[:max_trials], "choices": choices[:max_trials], "rewards": rewards[:max_trials]}


def load_episodes_from_csv_list(paths: Sequence[str], max_trials=256) -> list[dict]:
    """Load one episode per CSV."""
    return [load_episode_from_csv_2d_file(p, max_trials=max_trials) for p in paths]


# ------------------------------ Single-Env (2D) ------------------------------

class CSC2BehaviorEnv2D(gym.Env):
    """
    Deterministic 2D behavior environment driven by CSV-derived episodes.

    Obs: (k,2) angles (theta, phi) for the *current* trial's options.
    Reward: scalar reward for the *current* trial (from CSV).
    info['choice']: recorded choice for the *current* trial (from CSV).
    """
    metadata = {"render_modes": []}

    def __init__(self, *, angles: np.ndarray, choices: np.ndarray, rewards: np.ndarray):
        super().__init__()
        assert angles.ndim == 3 and angles.shape[-1] == 2, "angles must be (T,k,2)"
        T, k, _ = angles.shape
        assert choices.shape == (T,), "choices must be (T,)"
        assert rewards.shape == (T,), "rewards must be (T,)"

        self.angles  = np.asarray(angles, dtype=np.float32)
        self.choices = np.asarray(choices, dtype=np.int64)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.T, self.k = int(T), int(k)

        self.observation_space = spaces.Box(low=0.0, high=TAU, shape=(self.k, 2), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k)  # ignored by env; dataset dictates behavior
        self._t = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = self.angles[self._t]
        info = {"choice": int(self.choices[self._t])}
        return obs, info

    def step(self, action):
        # We expose reward for the *current* trial, then advance to the next trial.
        r = float(self.rewards[self._t])
        self._t += 1
        done = self._t >= self.T

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"choice": 0}  # dummy at terminal
        else:
            obs = self.angles[self._t]
            info = {"choice": int(self.choices[self._t])}

        truncated = False
        return obs, r, done, truncated, info

    def close(self):
        pass


# ---------------------------- Vectorized Wrapper -----------------------------

class SyncVectorEnvBehavior(SyncVectorEnv):
    """
    Sync vector wrapper that aggregates:
      - rewards: (B,) float32
      - info['choice']: (B,) int64
    """
    def reset(self, **kwargs):
        obs_list, choices = [], []
        for env in self.envs:
            o, info = env.reset(**kwargs)
            obs_list.append(o)
            choices.append(info.get("choice", 0))
        self.observations = np.stack(obs_list, axis=0)  # (B,k,2)
        self._terminateds = np.zeros((self.num_envs,), dtype=bool)
        self._truncateds  = np.zeros((self.num_envs,), dtype=bool)
        info = {"choice": np.asarray(choices, dtype=np.int64)}
        return self.observations, info

    def step_wait(self):
        obs_list, rewards, choices = [], [], []
        for i, a in enumerate(self._actions):
            o, r, term, trunc, info = self.envs[i].step(a)
            obs_list.append(o)
            rewards.append(r)
            self._terminateds[i] = term
            self._truncateds[i]  = trunc
            choices.append(info.get("choice", 0))
        self.observations = np.stack(obs_list, axis=0)
        rewards = np.asarray(rewards, dtype=np.float32)
        info = {"choice": np.asarray(choices, dtype=np.int64)}
        return self.observations, rewards, self._terminateds, self._truncateds, info


def make_csc2_behavior_vector_from_csvs(csv_paths: Sequence[str], max_trials: int=256*300) -> SyncVectorEnvBehavior:
    """
    Build a vectorized env where each CSV path is one episode/env.
    """
    episodes = load_episodes_from_csv_list(csv_paths, max_trials)
    env_fns = [partial(CSC2BehaviorEnv2D, **epi) for epi in episodes]
    return SyncVectorEnvBehavior(env_fns)


