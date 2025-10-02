import random
import time
from typing import Optional

from gymnasium import spaces
from functools import partial
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import math, colorsys, numpy as np, pygame, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('TkAgg')


TAU = 2 * math.pi


def _wrapped_gaussian_field(n: int, n_gauss: int = 8, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n_gauss = rng.integers(2, 8)
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    field = np.zeros((n, n), dtype=float)
    for _ in range(n_gauss):
        mu = rng.uniform(0, n, size=2)
        sigma_x = rng.uniform(n/15, n/6)
        sigma_y = rng.uniform(n/15, n/6)
        theta   = rng.uniform(0, 2 * np.pi)
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t],[sin_t,  cos_t]])
        D = np.diag([sigma_x**2, sigma_y**2])
        Sigma = R @ D @ R.T
        inv_Sigma = np.linalg.inv(Sigma)
        dx = (xs - mu[0] + n/2) % n - n/2
        dy = (ys - mu[1] + n/2) % n - n/2
        term_xx = inv_Sigma[0, 0] * dx * dx
        term_xy = 2 * inv_Sigma[0, 1] * dx * dy
        term_yy = inv_Sigma[1, 1] * dy * dy
        exponent = -0.5 * (term_xx + term_xy + term_yy)
        field += np.exp(exponent)
    return field

def _wrapped_gaussian_field_1d(n: int, n_gauss: int = 8, rng=None) -> np.ndarray:
    """Toroidal 1D mixture on a circle of length n."""
    rng = rng or np.random.default_rng()
    n_gauss = rng.integers(1, 4)
    x = np.arange(n)
    field = np.zeros(n, dtype=float)
    for _ in range(n_gauss):
        mu = rng.uniform(0, n)
        sigma = rng.uniform(n/20, n/8)
        # wrapped distance on circle
        dx = (x - mu + n/2) % n - n/2
        field += np.exp(-0.5 * (dx / sigma) ** 2)
    return field

# --------------------------------------------------------------------- #
class CSC2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            n: int = 36,
            k: int = 2,
            trials: int = 200,
            *,
            seed: Optional[int] = None,
            render: bool = False,
            win_size: int = 600,
            render_reward: bool = False,
            batched: bool = False,
            uniform_freq: bool = True,
            uniform_reward = False,
            one_d: bool = False,
            discrete_reward=False,
    ):
        super().__init__()

        # core parameters
        self.n, self.k, self.max_trials = n, k, trials
        self.rng = np.random.default_rng(seed)
        self.one_d = bool(one_d)

        # build reward & sampling fields
        if self.one_d:
            R = _wrapped_gaussian_field_1d(n, rng=self.rng)
            if discrete_reward:
                R = np.digitize(R, np.quantile(R, np.linspace(0.0, 1.0, 5)[1:-1]))
            self.reward_grid = (R - R.mean()) / R.std()
            if uniform_reward:
                self.reward_grid = np.zeros_like(self.reward_grid)
            F = _wrapped_gaussian_field_1d(n, rng=self.rng)
            if uniform_freq:
                F = np.ones_like(F)
            F = F / F.sum()
            self.freq_grid = F
            self._flat_probs = F  # length n
            # precompute theta table for 1D
            self._theta_table_1d = TAU * np.arange(n) / n  # (n,)
        else:
            R = _wrapped_gaussian_field(n, rng=self.rng)
            if discrete_reward:
                R = np.digitize(R, np.quantile(R, np.linspace(0.0, 1.0, 5)[1:-1]))
            self.reward_grid = (R - R.mean(keepdims=True)) / R.std(keepdims=True)
            F = _wrapped_gaussian_field(n, rng=self.rng)
            if uniform_freq:
                F = np.ones_like(F)
            F /= F.sum()
            self.freq_grid = F
            self._flat_probs = F.ravel()  # length n*n
            # precompute angle look-up table (θ, φ) for every cell
            idx = np.arange(n)
            theta = TAU * idx[:, None] / n  # (n,1)
            phi = TAU * idx[None, :] / n    # (1,n)
            self._angle_table = np.stack([np.tile(theta, (1, n)),
                                          np.tile(phi,   (n, 1))], -1)  # (n,n,2)

        # select mode and fix spaces before vectorisation
        self._batched = bool(batched)
        if self._batched:
            if self.one_d:
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(trials, k, 1), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0.0, high=TAU, shape=(trials, k, 2), dtype=np.float32
                )
            self.action_space = spaces.MultiDiscrete(np.full(trials, k, dtype=np.int64))
        else:
            if self.one_d:
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(k, 1), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0.0, high=TAU, shape=(k, 2), dtype=np.float32
                )
            self.action_space = spaces.Discrete(k)

        # episode state placeholders
        self._trial = 0
        self._idx = None          # sequential indices
        self._idx_batch = None    # batched indices
        self._last_obs = None

        # rendering parameters
        self.render_flag = render
        self.render_reward = render_reward
        self.win = win_size
        self._one_d_color = (200, 200, 200)  # constant color for 1D
        if render:
            pygame.init()
            self.scr = pygame.display.set_mode((win_size, win_size))
            pygame.display.set_caption("CSC-2 episode")

    # ------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------
    def _sample_indices(self):
        """k unique locations drawn from freq_grid (sequential mode)."""
        if self.one_d:
            flat_idx = self.rng.choice(self.n, size=self.k, replace=False, p=self._flat_probs)
            return flat_idx  # (k,)
        else:
            flat_idx = self.rng.choice(self.n * self.n, size=self.k, replace=False, p=self._flat_probs)
            return np.column_stack((flat_idx // self.n, flat_idx % self.n))  # (k,2)

    def _angles(self, idx):
        """Map indices to angles: 1D→theta; 2D→(theta,phi)."""
        return (TAU * idx / self.n).astype(np.float32)

    @staticmethod
    def _rgb_from_phi(phi):
        r, g, b = colorsys.hls_to_rgb(phi / TAU, 0.6, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None, batched: Optional[bool] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if batched is not None and bool(batched) != self._batched:
            raise ValueError("Cannot switch batched/sequential mode after construction.")
        self._trial = 0

        if self._batched:
            T, k, n = self.max_trials, self.k, self.n
            M = n if self.one_d else n * n
            gumbel = -np.log(-np.log(self.rng.random(size=(T, M))))
            scores = np.log(self._flat_probs) + gumbel  # (T,M)
            topk = np.argpartition(scores, -k, axis=1)[:, -k:]  # (T,k)
            if self.one_d:
                self._idx_batch = topk  # (T,k) indices 0..n-1
                obs = self._theta_table_1d[self._idx_batch][..., None]  # (T,k, 1)
            else:
                rows = topk // n
                cols = topk % n
                self._idx_batch = np.stack([rows, cols], axis=-1)              # (T,k,2)
                obs = self._angle_table[rows, cols].astype(np.float32)         # (T,k,2)
        else:
            self._idx = self._sample_indices()
            if self.one_d:
                obs = self._theta_table_1d[self._idx, None]                      # (k, 1)
                if self.render_flag:
                    self._draw_stimuli(obs)  # pass thetas for 1D
                    pygame.time.wait(400)
            else:
                rows, cols = self._idx[:, 0], self._idx[:, 1]
                obs = self._angle_table[rows, cols].astype(np.float32)         # (k,2)
                if self.render_flag:
                    self._draw_stimuli(obs)
                    pygame.time.wait(400)

        self._last_obs = obs
        return obs, {}

    def step(self, action):
        # -------------------------- batched ------------------------------ #
        if self._batched:
            acts = np.asarray(action, dtype=int)
            if acts.ndim == 1:
                acts = acts[None, :]
            B, T = acts.shape
            assert T == self.max_trials, "action vector length mismatch"
            assert ((0 <= acts) & (acts < self.k)).all(), "actions out of range"

            times = np.broadcast_to(np.arange(T), (B, T))
            if self.one_d:
                idx = self._idx_batch[times, acts]                 # (B,T)
                rewards = self.reward_grid[idx].astype(np.float32) # (B,T)
                obs_zeros = np.zeros((T, self.k), np.float32)
            else:
                idx = self._idx_batch[times, acts]                 # (B,T,2)
                rewards = self.reward_grid[idx[..., 0], idx[..., 1]].astype(np.float32)
                obs_zeros = np.zeros_like(self._angles(self._idx_batch), np.float32)

            return obs_zeros, rewards, True, False, {}

        # ------------------------ sequential ----------------------------- #
        a = int(action)
        assert 0 <= a < self.k, "action out of range"

        if self.one_d:
            reward = float(np.round(self.reward_grid[int(self._idx[a])], 1))
        else:
            reward = float(np.round(self.reward_grid[tuple(self._idx[a])], 1))

        if self.render_flag:
            self._draw_choice(a, reward)
            pygame.time.wait(200)

        self._trial += 1
        done = self._trial >= self.max_trials
        if done:
            obs = np.zeros(self.observation_space.shape, np.float32)
        else:
            self._idx = self._sample_indices()
            if self.one_d:
                obs = self._theta_table_1d[self._idx, None]
                if self.render_flag:
                    self._draw_stimuli(obs)
                    pygame.time.wait(300)
            else:
                obs = self._angles(self._idx)
                if self.render_flag:
                    self._draw_stimuli(obs)
                    pygame.time.wait(300)

        self._last_obs = obs
        return obs, reward, done, False, {}

    # ------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------
    def _event_pump(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _draw_stimuli(self, obs):
        """
        2D: obs shape (k,2) with (theta,phi), colored by phi.
        1D: obs is theta array (k,), colored constant.
        """
        self.scr.fill((0, 0, 0))
        cols = math.ceil(math.sqrt(self.k))
        rows = math.ceil(self.k / cols)
        cw, ch = self.win / cols, self.win / rows
        radius = 0.35 * min(cw, ch)
        for i in range(self.k):
            r, c = divmod(i, cols)
            cx, cy = int((c + 0.5) * cw), int((r + 0.5) * ch)
            pygame.draw.circle(self.scr, (180, 180, 180), (cx, cy), int(radius), 2)
            if self.one_d:
                theta = float(obs[i])
                color = self._one_d_color
            else:
                theta, phi = obs[i]
                color = self._rgb_from_phi(phi)
            dx, dy = radius * math.cos(theta), radius * math.sin(theta)
            pygame.draw.circle(self.scr, color, (int(cx + dx), int(cy + dy)), int(radius * 0.12))
        pygame.display.flip()
        self._event_pump()

    def _draw_choice(self, idx, reward_val=None):
        cols = math.ceil(math.sqrt(self.k))
        cw = self.win / cols
        radius = 0.35 * cw
        margin = 4
        r, c = divmod(idx, cols)
        cx, cy = int((c + 0.5) * cw), int((r + 0.5) * cw)
        size = int(2 * radius + 2 * margin)
        pygame.draw.rect(
            self.scr, (0, 255, 0),
            pygame.Rect(int(cx - radius - margin), int(cy - radius - margin), size, size),
            3
        )
        if reward_val is not None and self.render_reward:
            font = pygame.font.Font(None, int(self.win * 0.25))
            surf = font.render(str(reward_val), True, (255, 255, 255))
            rect = surf.get_rect(center=(self.win // 2, self.win // 2))
            self.scr.blit(surf, rect)
        pygame.display.flip()
        self._event_pump()

    def render(self):
        if self.render_flag and not self._batched and self._idx is not None:
            if self.one_d:
                thetas = self._theta_table_1d[self._idx]
                self._draw_stimuli(thetas)
            else:
                self._draw_stimuli(self._angles(self._idx))

    def close(self):
        if self.render_flag:
            pygame.quit()

    # ------------------------------------------------------------
    # optional diagnostics
    # ------------------------------------------------------------
    def show_fields(self):
        """Display reward and frequency structures (1D as lines, 2D as heatmaps)."""
        if self.one_d:
            fig, axs = plt.subplots(2, 1, figsize=(6, 4), constrained_layout=True)
            axs[0].plot(self.reward_grid)
            axs[0].set_title("Reward (1D, normalized)")
            axs[1].plot(self.freq_grid)
            axs[1].set_title("Frequency (1D, normalized)")
            plt.show()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            im0 = axs[0].imshow(self.reward_grid, origin="lower")
            axs[0].set_title("Reward (2D, normalized)")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            im1 = axs[1].imshow(self.freq_grid, origin="lower")
            axs[1].set_title("Frequency (2D, normalized)")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            plt.show()


class SyncVectorEnvVecReward(SyncVectorEnv):
    def step_wait(self):
        infos = []
        for i, a in enumerate(self._actions):
            obs, rew, terminated, truncated, info = self.envs[i].step(a)
            rew_arr = np.asarray(rew)
            if self._rewards is None or self._rewards.shape[1:] != rew_arr.shape:
                self._rewards = np.zeros((self.num_envs,) + rew_arr.shape, dtype=rew_arr.dtype)
                self._cumulative_rewards = np.zeros_like(self._rewards)
            self.observations[i] = obs
            self._rewards[i] = rew_arr
            self._cumulative_rewards[i] += rew_arr
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated
            infos.append(info)
        return (
            self.observations,
            self._rewards[:, 0, :],
            self._terminateds,
            self._truncateds,
            infos,
        )


def make_csc2_vector(
    num_envs: int,
    *,
    async_mode: bool = False,
    seeds: Optional[list[int]] = None,
    **env_kwargs,
):
    seeds = seeds or list([None] * num_envs)
    ctor = partial(CSC2Env, **env_kwargs)
    env_fns = [partial(ctor, seed=s) for s in seeds]
    if async_mode:
        raise NotImplementedError("Vector-rewards wrapper only provided for sync mode")
    VecCls = SyncVectorEnvVecReward if env_kwargs.get("batched", False) else SyncVectorEnv
    return VecCls(env_fns)


if __name__ == '__main__':
    # test environment
    env = CSC2Env(n=36, k=4, trials=50, render=False, seed=None, render_reward=False, one_d=True)
    env.show_fields()
    obs, _ = env.reset()
    while True:
        obs, r, done, _, _ = env.step(env.action_space.sample())
        if done:
            break
    env.close()