# mighty_rollout_buffer.py

from __future__ import annotations

import dill as pickle
import numpy as np
import torch

from mighty.mighty_replay.buffer import MightyBuffer


class MaxiBatch:
    def __init__(self, minibatches: list[RolloutBatch]):
        self.minibatches = minibatches

    # aggregated view ------------------------------------------------------

    def __getattribute__(self, name: str):
        if name in {
            "observations",
            "actions",
            "latents",  # ← NEW
            "rewards",
            "advantages",
            "returns",
            "episode_starts",
            "log_probs",
            "values",
        }:
            mbs = object.__getattribute__(self, "minibatches")
            if not mbs:
                return torch.tensor([])
            stacked = torch.stack(
                [
                    (
                        getattr(mb, name)
                        if getattr(mb, name) is not None
                        else torch.zeros_like(getattr(mbs[0], "actions"))
                    )
                    for mb in mbs
                ],  # type: ignore
                dim=0,
            )
            feat_shape = (
                getattr(mbs[0], name).shape[1:]
                if getattr(mbs[0], name) is not None
                else ()
            )
            return stacked.reshape(-1, *feat_shape)
        return object.__getattribute__(self, name)

    # trivial helpers ------------------------------------------------------

    @property
    def size(self) -> int:
        return sum(len(mb) for mb in self.minibatches)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from self.minibatches


class RolloutBatch:
    """A contiguous slice of experience – now stores the latent ``z`` too."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        latents: np.ndarray | None = None,  # ← NEW (optional for discrete)
        rewards: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        episode_starts: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        device: torch.device | str = "cpu",
    ):
        self.device = device

        obs_t = torch.from_numpy(observations.astype(np.float32))
        act_t = torch.from_numpy(actions.astype(np.float32))
        lat_t = (
            torch.from_numpy(latents.astype(np.float32))
            if latents is not None
            else None
        )  # may stay None for discrete
        rew_t = torch.from_numpy(rewards.astype(np.float32))
        adv_t = torch.from_numpy(advantages.astype(np.float32))
        ret_t = torch.from_numpy(returns.astype(np.float32))
        eps_t = torch.from_numpy(episode_starts.astype(np.float32))
        logp_t = torch.from_numpy(log_probs.astype(np.float32))
        val_t = torch.from_numpy(values.astype(np.float32))

        # Promote obs from [n_envs, obs_dim] → [1, n_envs, obs_dim] if needed
        if obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.dim() < 2:
            raise RuntimeError(
                f"RolloutBatch: `observations` must be ≥2‑D, got {obs_t.shape}"
            )

        def _promote(x: torch.Tensor | None, name: str):
            if x is None:
                return None
            if x.dim() == 1:  # (n_envs,) → (1, n_envs)
                return x.unsqueeze(0)
            elif x.dim() == 2:  # (timesteps, n_envs) - already correct
                return x
            elif x.dim() == 3 and name in [
                "actions",
                "observations",
            ]:  # (timesteps, n_envs, features)
                return x
            else:
                raise RuntimeError(f"Unexpected shape for {name}: {x.shape}")

        act_t = _promote(act_t, "actions")
        lat_t = _promote(lat_t, "latents")
        rew_t = _promote(rew_t, "rewards")
        adv_t = _promote(adv_t, "advantages")
        ret_t = _promote(ret_t, "returns")
        eps_t = _promote(eps_t, "episode_starts")
        logp_t = _promote(logp_t, "log_probs")
        val_t = _promote(val_t, "values")

        # Move to device ---------------------------------------------------
        self.observations = obs_t.to(self.device)
        self.actions = act_t.to(self.device)
        self.latents = lat_t.to(self.device) if lat_t is not None else None
        self.rewards = rew_t.to(self.device)
        self.advantages = adv_t.to(self.device)
        self.returns = ret_t.to(self.device)
        self.episode_starts = eps_t.to(self.device)
        self.log_probs = logp_t.to(self.device)
        self.values = val_t.to(self.device)

    # Basic helpers --------------------------------------------------------

    @property
    def size(self) -> int:
        return self.observations.shape[0]

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from zip(
            self.observations,
            self.actions,
            self.latents if self.latents is not None else [None] * len(self),
            self.rewards,
            self.advantages,
            self.returns,
            self.episode_starts,
            self.log_probs,
            self.values,
            strict=False,
        )


class MightyRolloutBuffer(MightyBuffer):
    """
    Pre-allocated rollout buffer (no repeated concat).
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        act_dim,
        device: torch.device | str = "cpu",
        *,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        discrete_action: bool = False,
        use_latents: bool = False,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.discrete_action = discrete_action
        self.use_latents = use_latents  # Store for later use
        self.rng = np.random.default_rng()

        # Shapes -----------------------------------------------------------
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)

        def zeros(shape):
            return torch.zeros(shape, dtype=torch.float32, device=device)

        self.observations = zeros((buffer_size, n_envs, *obs_shape))

        if discrete_action:
            self.actions = zeros((buffer_size, n_envs))
            self.latents = None  # not used
        else:
            self.actions = zeros((buffer_size, n_envs, act_dim))

            if use_latents:
                self.latents = zeros((buffer_size, n_envs, act_dim))
            else:
                self.latents = None

        self.rewards = zeros((buffer_size, n_envs))
        self.advantages = zeros((buffer_size, n_envs))
        self.returns = zeros((buffer_size, n_envs))
        self.episode_starts = zeros((buffer_size, n_envs))
        self.log_probs = zeros((buffer_size, n_envs))
        self.values = zeros((buffer_size, n_envs))

        self.pos = 0

    def reset(self) -> None:
        # Just zero out pos; no need to re-allocate
        self.pos = 0

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,  # shape = [n_envs] or [n_envs, 1]
        dones: np.ndarray,  # shape = [n_envs]
    ) -> None:
        if self.pos == 0:
            return

        # 1) Turn last_values into a 1‐D tensor of shape [n_envs]
        lv = last_values.clone().to(self.device).reshape(-1)  # → [n_envs]

        # 2) Turn the numpy dones (0/1) into float tensor on device, shape [n_envs]
        dones_t = (
            torch.from_numpy(dones.astype(np.float32)).to(self.device).reshape(-1)
        )  # → [n_envs]

        T = self.pos  # number of filled "time‐steps" in buffer
        N = self.n_envs  # number of parallel envs

        # 3) Get slices (these are views, but we'll create new tensors for advantages)
        rew_slice = self.rewards[:T]  # (T × n_envs)
        val_slice = self.values[:T]  # (T × n_envs)
        eps_slice = self.episode_starts[:T]  # (T × n_envs)

        # 4) Create NEW tensors for advantages and returns (not views!)
        advantages = torch.zeros_like(rew_slice)  # [T, n_envs]

        # 5) Initialize the last‐step GAE "accumulator" to zero for each env
        last_gae = torch.zeros(N, device=self.device)  # [n_envs]

        # 6) Walk backwards over time steps
        for step in reversed(range(T)):
            if step == T - 1:
                # On the very last (most recent) buffer row:
                next_non_term = 1.0 - dones_t  # [n_envs], 0 if done, 1 otherwise
                next_val = lv  # bootstrap from V(sₜ₊₁)
            else:
                # On intermediate steps, look at "episode_starts" for whether step+1 was a new episode
                next_non_term = 1.0 - eps_slice[step + 1]  # [n_envs]
                next_val = val_slice[step + 1]  # [n_envs]

            r_t = rew_slice[step]  # shape = [n_envs]
            r_t[r_t.isnan()] = 0.0  # treat NaN rewards (from autoreset) as zero
            v_t = val_slice[step]  # shape = [n_envs]
            v_t[v_t.isnan()] = 0.0  # treat NaN values (from autoreset) as zero

            # standard TD residual
            delta = r_t + self.gamma * next_val * next_non_term - v_t  # [n_envs]

            # recursive GAE
            last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_term * last_gae
            )  # [n_envs]

            # store into NEW advantage tensor (not a view!)
            advantages[step] = last_gae

        # 7) Compute returns and write back to buffer
        returns = advantages + val_slice
        self.advantages[:T] = advantages
        self.returns[:T] = returns

    #     self.pos += n_steps
    def add(self, rollout_batch: RolloutBatch, _=None):
        rb = rollout_batch  # alias
        n_steps = rb.observations.shape[0]
        if self.pos + n_steps > self.buffer_size:
            raise RuntimeError("Buffer overflow – increase buffer_size.")

        sl = slice(self.pos, self.pos + n_steps)
        self.observations[sl] = rb.observations
        self.actions[sl] = rb.actions
        if self.latents is not None and rb.latents is not None:
            self.latents[sl] = rb.latents  # ← copy latents
        self.rewards[sl] = rb.rewards
        self.advantages[sl] = rb.advantages
        self.returns[sl] = rb.returns
        self.episode_starts[sl] = rb.episode_starts
        self.log_probs[sl] = rb.log_probs  # keep original quirk
        self.values[sl] = rb.values
        self.pos += n_steps

    #     return MaxiBatch(mini_batches)
    def sample(self, batch_size: int) -> MaxiBatch:
        if self.pos == 0:
            return MaxiBatch([])

        T, N = self.pos, self.n_envs
        total = T * N
        if total < batch_size:
            return MaxiBatch([])

        def _flat(t: torch.Tensor | None):
            if t is None:
                return None
            return t[:T].reshape(total, *t.shape[2:]).cpu().numpy()

        obs_f = _flat(self.observations)
        acts_f = _flat(self.actions)
        lats_f = _flat(self.latents) if self.latents is not None else None
        rew_f = _flat(self.rewards)
        adv_f = _flat(self.advantages)
        ret_f = _flat(self.returns)
        eps_f = _flat(self.episode_starts)
        logp_f = _flat(self.log_probs)
        val_f = _flat(self.values)

        perm = self.rng.permutation(total)
        perm = perm[: (total // batch_size) * batch_size].reshape(-1, batch_size)

        minibatches: list[RolloutBatch] = []
        for inds in perm:
            minibatches.append(
                RolloutBatch(
                    observations=obs_f[inds],
                    actions=acts_f[inds],
                    latents=None if lats_f is None else lats_f[inds],
                    rewards=rew_f[inds],
                    advantages=adv_f[inds],
                    returns=ret_f[inds],
                    episode_starts=eps_f[inds],
                    log_probs=logp_f[inds],
                    values=val_f[inds],
                )
            )

        return MaxiBatch(minibatches)

    def __len__(self) -> int:
        return self.pos * self.n_envs

    def __bool__(self) -> bool:
        return self.pos > 0

    def save(self, filename="buffer.pkl") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)
