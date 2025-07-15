import time
from typing import Optional

from gymnasium import spaces
from functools import partial
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import math, colorsys, numpy as np, pygame, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

TAU = 2 * math.pi


def _wrapped_gaussian_field(n: int, n_gauss: int = 4, rng=None) -> np.ndarray:
    """
    Generate an n×n toroidal field composed of n_gauss 2D Gaussians
    with random anisotropic covariances, using ASCII-only names.
    """
    rng = rng or np.random.default_rng()
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    field = np.zeros((n, n), dtype=float)

    for _ in range(n_gauss):
        # random center
        mu = rng.uniform(0, n, size=2)

        # random anisotropic scales
        sigma_x = rng.uniform(n/15, n/3)
        sigma_y = rng.uniform(n/15, n/3)
        theta   = rng.uniform(0, 2 * np.pi)

        # build covariance Sigma = R · diag(sigma_x^2, sigma_y^2) · R^T
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])
        D = np.diag([sigma_x**2, sigma_y**2])
        Sigma = R @ D @ R.T

        # inverse covariance
        inv_Sigma = np.linalg.inv(Sigma)

        # toroidal (wrapped) distances from mean
        dx = (xs - mu[0] + n/2) % n - n/2
        dy = (ys - mu[1] + n/2) % n - n/2

        # expand the quadratic form manually:
        term_xx = inv_Sigma[0, 0] * dx * dx
        term_xy = 2 * inv_Sigma[0, 1] * dx * dy
        term_yy = inv_Sigma[1, 1] * dy * dy
        exponent = -0.5 * (term_xx + term_xy + term_yy)

        # add the Gaussian bump
        field += np.exp(exponent)

    return field

# --------------------------------------------------------------------- #
class CSC2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            n: int = 36,
            k: int = 4,
            trials: int = 200,
            *,
            seed: Optional[int] = None,
            render: bool = False,
            win_size: int = 600,
            render_reward: bool = False,
            batched: bool = False,
    ):
        """
        Initialize CSC2Env.

        Parameters
        ----------
        n             : int
            Grid size (n × n).
        k             : int
            Number of options per trial.
        trials        : int
            Number of trials per episode (T).
        seed          : Optional[int]
            RNG seed.
        render        : bool
            Whether to enable pygame rendering.
        win_size      : int
            Window size (pixels).
        render_reward : bool
            Whether to overlay reward numerals.
        batched       : bool
            If True, operate in batched mode (reset→(T,k,2), step→(T,)).
        """
        super().__init__()

        # core parameters
        self.n, self.k, self.max_trials = n, k, trials
        self.rng = np.random.default_rng(seed)

        # build reward & sampling fields
        R = _wrapped_gaussian_field(n, rng=self.rng)
        self.reward_grid = np.digitize(
            R,
            np.quantile(R, np.linspace(0.0, 1.0, 5)[1:-1])
        )
        F = _wrapped_gaussian_field(n, rng=self.rng)
        F /= F.sum()
        self.freq_grid = F

        # precompute flat probability vector for sampling
        self._flat_probs = F.ravel()  # shape (n*n,)

        # precompute angle look-up table (θ, φ) for every cell
        idx = np.arange(n)
        theta = TAU * idx[:, None] / n  # (n,1)
        phi = TAU * idx[None, :] / n  # (1,n)
        self._angle_table = np.stack([np.tile(theta, (1, n)), np.tile(phi, (n, 1))], -1)  # (n,n,2)

        # select mode and fix spaces before vectorisation
        self._batched = bool(batched)
        if self._batched:
            self.observation_space = spaces.Box(
                low=0.0, high=TAU, shape=(trials, k, 2), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete(
                np.full(trials, k, dtype=np.int64)
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=TAU, shape=(k, 2), dtype=np.float32
            )
            self.action_space = spaces.Discrete(k)

        # episode state placeholders
        self._trial = 0
        self._idx = None  # for sequential mode
        self._idx_batch = None  # for batched mode

        # rendering parameters
        self.render_flag = render
        self.render_reward = render_reward
        self.win = win_size
        if render:
            pygame.init()
            self.scr = pygame.display.set_mode((win_size, win_size))
            pygame.display.set_caption("CSC-2 episode")

    # ------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------
    def _sample_indices(self):
        """k unique locations drawn from `freq_grid` (sequential mode)."""
        flat_idx = self.rng.choice(
            self.n * self.n, size=self.k, replace=False, p=self._flat_probs
        )
        return np.column_stack((flat_idx // self.n, flat_idx % self.n))

    def _angles(self, idx):
        return (TAU * idx / self.n).astype(np.float32)

    @staticmethod
    def _rgb_from_phi(phi):
        r, g, b = colorsys.hls_to_rgb(phi / TAU, 0.6, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None, batched: Optional[bool] = None):
        """
        Initialise an episode. In batched mode, pre-sample all trials at once.
        """
        super().reset(seed=seed)
        # reseed RNG if requested
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # disallow switching mode after construction
        if batched is not None and bool(batched) != self._batched:
            raise ValueError("Cannot switch batched/sequential mode after construction.")

        # reset sequential counter
        self._trial = 0

        if self._batched:
            # vectorised sample-without-replacement across all T trials
            T, k, n = self.max_trials, self.k, self.n
            M = n * n

            # draw Gumbel noise and form perturbed log-probs
            gumbel = -np.log(-np.log(self.rng.random(size=(T, M))))
            scores = np.log(self._flat_probs) + gumbel  # shape (T, M)

            # take top-k indices per trial (unordered)
            topk = np.argpartition(scores, -k, axis=1)[:, -k:]  # (T, k)

            rows = topk // n  # (T, k)
            cols = topk % n  # (T, k)
            self._idx_batch = np.stack([rows, cols], axis=-1)  # (T, k, 2)

            # lookup angles via precomputed table → (T, k, 2)
            obs = self._angle_table[rows, cols]
        else:
            # one trial only
            self._idx = self._sample_indices()  # (k, 2)
            rows, cols = self._idx[:, 0], self._idx[:, 1]
            obs = self._angle_table[rows, cols]  # (k, 2)

            if self.render_flag:
                self._draw_stimuli(obs)
                pygame.time.wait(400)

        return obs, {}

    def step(self, action):
        # -------------------------- batched ------------------------------ #
        if self._batched:
            acts = np.asarray(action, dtype=int)
            if acts.ndim == 1:                    # (T,)
                acts = acts[None, :]              # --> (1,T)
            B, T = acts.shape
            assert T == self.max_trials, "action vector length mismatch"
            assert ((0 <= acts) & (acts < self.k)).all(), "actions out of range"

            # gather rewards
            times = np.broadcast_to(np.arange(T), (B, T))
            idx = self._idx_batch[times, acts]          # (B,T,2)
            rewards = self.reward_grid[idx[..., 0], idx[..., 1]].astype(int)  # (B,T)

            obs_zeros = np.zeros_like(self._angles(self._idx_batch), np.float32)
            return obs_zeros, rewards, True, False, {}

        # ------------------------ sequential ----------------------------- #
        a = int(action)
        assert 0 <= a < self.k, "action out of range"
        reward = int(self.reward_grid[tuple(self._idx[a])])

        if self.render_flag:
            self._draw_choice(a, reward)
            pygame.time.wait(200)

        self._trial += 1
        done = self._trial >= self.max_trials
        if done:
            obs = np.zeros((self.k, 2), np.float32)
        else:
            self._idx = self._sample_indices()
            obs = self._angles(self._idx)
            if self.render_flag:
                self._draw_stimuli(obs)
                pygame.time.wait(300)

        return obs, reward, done, False, {}

    # ------------------------------------------------------------
    # rendering helpers (unchanged)
    # ------------------------------------------------------------
    def _event_pump(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _draw_stimuli(self, obs):
        self.scr.fill((0, 0, 0))
        cols = math.ceil(math.sqrt(self.k))
        rows = math.ceil(self.k / cols)
        cw, ch = self.win / cols, self.win / rows
        radius = 0.35 * min(cw, ch)
        for i in range(self.k):
            r, c = divmod(i, cols)
            cx, cy = int((c + 0.5) * cw), int((r + 0.5) * ch)
            pygame.draw.circle(self.scr, (180, 180, 180), (cx, cy), int(radius), 2)
            theta, phi = obs[i]
            dx, dy = radius * math.cos(theta), radius * math.sin(theta)
            pygame.draw.circle(
                self.scr, self._rgb_from_phi(phi),
                (int(cx + dx), int(cy + dy)), int(radius * 0.12)
            )
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
            self._draw_stimuli(self._angles(self._idx))

    def close(self):
        if self.render_flag:
            pygame.quit()



class SyncVectorEnvVecReward(SyncVectorEnv):
    """Exactly like SyncVectorEnv, but ``reward`` may be an array.

    The first time we see a reward, we allocate the big _rewards
    buffer with shape::

        (num_envs, *reward.shape)

    Subsequent steps simply copy the array.
    """

    def step_wait(self):
        infos = []

        for i, a in enumerate(self._actions):
            obs, rew, terminated, truncated, info = self.envs[i].step(a)

            # --- lazy allocation of the reward buffer --------------------
            rew_arr = np.asarray(rew)
            if self._rewards is None or self._rewards.shape[1:] != rew_arr.shape:
                self._rewards = np.zeros((self.num_envs,) + rew_arr.shape,
                                         dtype=rew_arr.dtype)
                self._cumulative_rewards = np.zeros_like(self._rewards)

            # -------------------------------------------------------------
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
    """Factory for a vectorised CSC-2 environment.

    *Sequential* episodes use Gymnasium’s stock wrappers.
    *Batched* episodes (reward vectors) use the custom wrapper above.
    """
    seeds = seeds or list(range(num_envs))
    ctor = partial(CSC2Env, **env_kwargs)
    env_fns = [partial(ctor, seed=s) for s in seeds]

    if async_mode:
        raise NotImplementedError("Vector-rewards wrapper only provided for sync mode")

    # choose wrapper based on mode
    if env_kwargs.get("batched", False):
        VecCls = SyncVectorEnvVecReward
    else:
        VecCls = SyncVectorEnv

    return VecCls(env_fns)


if __name__ == '__main__':
    # test environment
    env = CSC2Env(n=24, k=4, trials=50, render=True, seed=None, render_reward=True)
    env.show_fields()  # optional heat-maps
    obs, _ = env.reset()
    while True:
        obs, r, done, _, _ = env.step(env.action_space.sample())
        if done:
            break
    env.close()