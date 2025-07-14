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
    """
    Color-Shape-Choice (CSC-2) environment.

    **Sequential mode** (default)
        reset()  ->  (k, 2) observation          # one trial
        step(a) ->  reward (int), next obs, ...

    **Batched mode**
        reset(batched=True)
            ->  (T, k, 2) observation            # T = max_trials
        step(action_vec)                         # len(action_vec) == T
            ->  reward_vec (T,)  – then episode terminates
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------- constructor -------------------------- #
    def __init__(
        self,
        n: int = 24,
        k: int = 4,
        trials: int = 200,
        *,
        seed: Optional[int] = None,
        render: bool = False,
        win_size: int = 600,
        render_reward: bool = False,
        batched: bool = False,
    ):
        super().__init__()

        # grid size, items per trial, episode length
        self.n, self.k, self.max_trials = n, k, trials
        self.rng = np.random.default_rng(seed)

        # reward and sampling fields
        R = _wrapped_gaussian_field(n, rng=self.rng)
        self.reward_grid = np.digitize(R, np.quantile(R, np.linspace(0.0, 1.0, 5)[1:-1]))
        F = _wrapped_gaussian_field(n, rng=self.rng)
        self.freq_grid = F / F.sum()

        # spaces
        self.observation_space = spaces.Box(0.0, TAU, (k, 2), np.float32)
        self.action_space = spaces.Discrete(k)

        # episode state
        self._trial = 0                          # sequential counter
        self._idx: Optional[np.ndarray] = None   # (k,2) indices for current trial
        self._batched = batched
        self._idx_batch: Optional[np.ndarray] = None  # (T,k,2) for batched

        # rendering
        self.render_flag = render
        self.render_reward = render_reward
        self.win = win_size
        if render:
            pygame.init()
            self.scr = pygame.display.set_mode((win_size, win_size))
            pygame.display.set_caption("CSC-2 episode")

    # ------------------------- visual helpers ----------------------- #
    def show_fields(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        im1 = ax1.imshow(self.reward_grid, cmap="viridis", vmin=0, vmax=3)
        fig.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
        ax1.set_title("Reward levels")
        im2 = ax2.imshow(self.freq_grid, cmap="magma")
        fig.colorbar(im2, ax=ax2)
        ax2.set_title("Sampling frequency")
        plt.tight_layout()
        plt.show()

    # --------------------------- internals -------------------------- #
    def _sample_indices(self) -> np.ndarray:
        """Sample `k` unique (row, col) locations according to freq_grid."""
        flat = self.rng.choice(self.n * self.n, size=self.k, replace=False, p=self.freq_grid.ravel())
        return np.column_stack(np.unravel_index(flat, (self.n, self.n)))

    def _angles(self, idx: np.ndarray) -> np.ndarray:
        """Convert indices to (θ, φ) on the torus. Works on (k,2) or (T,k,2)."""
        return (TAU * idx / self.n).astype(np.float32)

    @staticmethod
    def _rgb_from_phi(phi: float) -> tuple[int, int, int]:
        r, g, b = colorsys.hls_to_rgb(phi / TAU, 0.6, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    # ------------------------------ draw ---------------------------- #
    def _event_pump(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _draw_stimuli(self, obs: np.ndarray):
        """
        Draw k grey circles with coloured dots.
        Only called in sequential mode.
        """
        self.scr.fill((0, 0, 0))
        k, win = self.k, self.win
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k / cols)
        cw, ch = win / cols, win / rows
        radius = 0.35 * min(cw, ch)
        grey = (180, 180, 180)

        for i in range(k):
            r, c = divmod(i, cols)
            cx, cy = int((c + 0.5) * cw), int((r + 0.5) * ch)
            pygame.draw.circle(self.scr, grey, (cx, cy), int(radius), 2)
            theta, phi = obs[i]
            dx, dy = radius * math.cos(theta), radius * math.sin(theta)
            pygame.draw.circle(
                self.scr,
                self._rgb_from_phi(phi),
                (int(cx + dx), int(cy + dy)),
                int(radius * 0.12),
            )

        pygame.display.flip()
        self._event_pump()

    def _draw_choice(self, choice_idx: int, reward_val: int | None = None):
        """Green box around chosen item; optional reward numeral."""
        k, win = self.k, self.win
        cols = math.ceil(math.sqrt(k))
        rows = math.ceil(k / cols)
        cw, ch = win / cols, win / rows
        radius = 0.35 * min(cw, ch)
        margin = 4

        r, c = divmod(choice_idx, cols)
        cx, cy = int((c + 0.5) * cw), int((r + 0.5) * ch)
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

    # --------------------------- logging ---------------------------- #
    def _log_trial(self, obs: np.ndarray):
        """Print shape / colour indices and rewards (sequential mode)."""
        rewards = self.reward_grid[self._idx[:, 0], self._idx[:, 1]]
        for i, (ang, idx, rwd) in enumerate(zip(obs, self._idx, rewards)):
            theta, phi = ang
            si, ci = idx
            print(
                f"Item {i}: θ={theta:.3f}  φ={phi:.3f} | "
                f"shape={si} colour={ci} | reward={rwd}"
            )
        print("-" * 60, flush=True)

    # ---------------------------- Gym API --------------------------- #
    def reset(self, *, seed: Optional[int] = None, options=None, batched: bool = None):
        """
        Initialise an episode.

        Parameters
        ----------
        batched : None or bool
            • False (default) – sequential mode.
            • True            – batched mode (all trials returned at once).
            - None use current set value
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if batched is not None:
            self._batched = bool(batched)
        self._trial = 0

        if self._batched:
            # full episode sampled up-front
            T = self.max_trials
            self._idx_batch = np.stack([self._sample_indices() for _ in range(T)], axis=0)  # (T,k,2)
            obs = self._angles(self._idx_batch)                                             # (T,k,2)

            # update observation_space to reflect batch dimension
            self.observation_space = spaces.Box(0.0, TAU, (T, self.k, 2), np.float32)
        else:
            self._idx = self._sample_indices()                 # (k,2)
            obs = self._angles(self._idx)                      # (k,2)
            self.observation_space = spaces.Box(0.0, TAU, (self.k, 2), np.float32)

            if self.render_flag:
                self._log_trial(obs)
                self._draw_stimuli(obs)
                pygame.time.wait(400)  # 0.40 s before first action

        return obs, {}

    def step(self, action):
        """
        Take an action (or actions).

        Sequential mode
        ---------------
        action : int or length-1 array
            Index 0 ≤ a < k of the chosen option.

        Batched mode
        ------------
        action : array-like, shape (T,)
            T = max_trials; a[t] is the choice for trial t.
        """
        # -------------------------- batched ------------------------- #
        if self._batched:
            acts = np.asarray(action, dtype=int).ravel()
            assert acts.shape == (self.max_trials,), "action vector length mismatch"
            assert ((0 <= acts) & (acts < self.k)).all(), "actions out of range"

            idx = self._idx_batch[np.arange(self.max_trials), acts]   # (T,2)
            rewards = self.reward_grid[idx[:, 0], idx[:, 1]].astype(int)  # (T,)

            obs_zeros = np.zeros_like(self._angles(self._idx_batch), dtype=np.float32)
            terminated, truncated = True, False
            return obs_zeros, rewards, terminated, truncated, {}

        # ------------------------ sequential ------------------------ #
        action_int = int(action)          # allow length-1 arrays from vector envs
        assert 0 <= action_int < self.k, "action out of range"

        if self.render_flag:
            reward_val = int(self.reward_grid[tuple(self._idx[action_int])])
            self._draw_choice(action_int, reward_val)
            pygame.time.wait(200)

        reward = int(self.reward_grid[tuple(self._idx[action_int])])

        # advance trial counter
        self._trial += 1
        terminated = self._trial >= self.max_trials
        truncated = False

        if terminated:
            obs_next = np.zeros((self.k, 2), dtype=np.float32)
        else:
            self._idx = self._sample_indices()
            obs_next = self._angles(self._idx)
            if self.render_flag:
                self._log_trial(obs_next)
                self._draw_stimuli(obs_next)
                pygame.time.wait(300)

        return obs_next, reward, terminated, truncated, {}

    # ----------------------------- misc ----------------------------- #
    def render(self):
        if self.render_flag and not self._batched and self._idx is not None:
            self._draw_stimuli(self._angles(self._idx))

    def close(self):
        if self.render_flag:
            pygame.quit()


def make_csc2_vector(num_envs: int,
                     async_mode: bool = False,
                     seeds: list[int] | None = None,
                     **env_kwargs):
    """
    Return a Gymnasium VectorEnv of `num_envs` independent CSC2 episodes.

    Parameters
    ----------
    num_envs   : number of parallel episodes
    async_mode : use AsyncVectorEnv (True) or SyncVectorEnv (False)
    seeds      : optional list of per-env seeds (len == num_envs)
    env_kwargs : forwarded to CSC2Env(...)

    Example
    -------
    vec = make_csc2_vector(8, n=24, k=4, trials=200)
    obs, info = vec.reset()
    obs, rew, term, trunc, info = vec.step(vec.action_space.sample())
    """
    seeds = seeds or list(range(num_envs))
    ctor   = partial(CSC2Env, **env_kwargs)
    env_fns = [partial(ctor, seed=s) for s in seeds]
    VecCls  = AsyncVectorEnv if async_mode else SyncVectorEnv
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