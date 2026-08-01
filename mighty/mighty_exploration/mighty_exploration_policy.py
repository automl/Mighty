"""Mighty Exploration Policy."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical, Normal



def sample_nondeterministic_logprobs(
    z: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    tanh_squash: bool = False,
) -> torch.Tensor:
    """
    Compute log-prob of a Gaussian sample z ~ N(mean, exp(log_std)).

    If tanh_squash=True, applies the change-of-variables correction for
    the squashed action a = tanh(z):

        log π(a) = log N(z | mean, std) − ∑ log(1 − tanh(z)²)

    Both old and new log-probs must use the same value of tanh_squash so
    the correction cancels correctly in the PPO importance-sampling ratio.
    """
    std = torch.exp(log_std)  # [batch, action_dim]
    dist = Normal(mean, std)
    # base Gaussian log‐prob of z
    log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)  # [batch, 1]

    if tanh_squash:
        # subtract the ∑_i log(d tanh/dz_i) = ∑ log(1 - tanh(z)^2)
        eps = 1e-6
        log_correction = torch.log(1.0 - torch.tanh(z).pow(2) + eps).sum(
            dim=-1, keepdim=True
        )  # [batch, 1]
        return log_pz - log_correction

    return log_pz


class MightyExplorationPolicy:
    """Generic Exploration Policy Interface.

    Now supports:
      - Discrete: `model(state)` → logits → Categorical
      - Continuous (squashed-Gaussian): `model(state)` → (action, z, mean, log_std)
      - Continuous (Standard PPO): `model(state)` → (action, mean, log_std)
      - Continuous (legacy): `model(state)` → (mean, std)
    """

    def __init__(
        self,
        algo,
        model,
        discrete=False,
    ) -> None:
        """
        :param algo:    Algorithm name (e.g. "ppo", "sac", etc.)
        :param model:   The policy network (any nn.Module)
        :param discrete: True if action-space is discrete
        """
        self.rng = np.random.default_rng()
        self.algo = algo
        self.model = model
        self.discrete = discrete

        # Check which action sampling to use
        if self.algo == "q":
            self.sample_action = self.sample_func_q
        else:
            self.sample_action = self.sample_func_logits

    def seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def sample_func_q(self, state_array):
        """
        Q-learning branch:
          • state_np: np.ndarray of shape [batch, obs_dim]
          • model(state) returns Q-values: tensor [batch, n_actions]
        We choose action = argmax(Q), and also return the full Q‐vector.
        """
        state = torch.as_tensor(state_array, dtype=torch.float32)
        qs = self.model(state)  # [batch, n_actions]
        # Choose greedy action
        action = torch.argmax(qs, dim=1)  # [batch]
        return action.detach().cpu().numpy(), qs  # action_np, Q‐vector

    def sample_func_logits(self, state_array):
        """
        state_np: np.ndarray of shape [batch, obs_dim]
        Returns: (action_tensor, log_prob_tensor)
        """
        state = torch.as_tensor(state_array, dtype=torch.float32)

        # ─── Discrete action branch ─────────────────────────────────────────
        if self.discrete:
            logits = self.model(state)  # [batch, n_actions]
            dist = Categorical(logits=logits)
            action = dist.sample()  # [batch]
            log_prob = dist.log_prob(action)  # [batch]
            return action.detach().cpu().numpy(), log_prob

        # ─── Continuous action branches ─────────────────────────────────────
        out = self.model(state)

        # NEW: Handle 3-tuple (Standard PPO)
        if isinstance(out, tuple) and len(out) == 3:
            action, mean, log_std = out
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)  # Direct log prob
            return action.detach().cpu().numpy(), log_prob

        # ─── Continuous squashed‐Gaussian (4‐tuple) ──────────────────────────
        elif isinstance(out, tuple) and len(out) == 4:
            action = out[0]  # [batch, action_dim]
            log_prob = sample_nondeterministic_logprobs(
                z=out[1], mean=out[2], log_std=out[3],
                tanh_squash=getattr(self.model, "tanh_squash", False),
            )
            return action.detach().cpu().numpy(), log_prob

        # ─── Legacy continuous branch (model returns (mean, std)) ────────────
        elif isinstance(out, tuple) and len(out) == 2:
            mean, std = out  # both [batch, action_dim]
            dist = Normal(mean, std)
            z = dist.rsample()  # [batch, action_dim]
            action = torch.tanh(z)  # [batch, action_dim]

            # 3a) log_pz = ∑ᵢ log N(zᵢ; μᵢ, σᵢ)
            log_pz = dist.log_prob(z).sum(dim=-1)  # [batch]

            # 3b) tanh‐correction
            eps = 1e-6
            log_correction = torch.log(1.0 - action.pow(2) + eps).sum(dim=-1)  # [batch]

            log_prob = log_pz - log_correction  # [batch]
            return action.detach().cpu().numpy(), log_prob

        # ─── Fallback: if model(state) returns a Distribution ────────────────
        elif isinstance(out, torch.distributions.Distribution):
            dist = out  # user returned a Distribution
            action = dist.sample()  # [batch]
            log_prob = dist.log_prob(action)  # [batch]
            return action.detach().cpu().numpy(), log_prob

        # ─── Otherwise, we don't know how to sample ─────────────────────────
        else:
            raise RuntimeError(
                "MightyExplorationPolicy: cannot interpret model(state) output of type "
                f"{type(out)}"
            )

    def __call__(self, s, return_logp=False, metrics=None, evaluate=False):
        """Get action.

        :param s: state
        :param return_logp: return logprobs
        :param metrics: current metric dict
        :param eval: eval mode
        :return: action or (action, logprobs)
        """
        if metrics is None:
            metrics = {}
        if evaluate:
            action, logprobs = self.sample_action(s)
            output = (action, logprobs) if return_logp else action
        else:
            output = self.explore(s, return_logp, metrics)

        return output

    def explore(self, s, return_logp, metrics=None):
        """Explore.

        :param s: state
        :param return_logp: return logprobs
        :param _: not used
        :return: action or (action, logprobs)
        """
        action, logprobs = self.explore_func(s)
        return (action, logprobs) if return_logp else action

    def explore_func(self, s):
        """Explore function."""
        raise NotImplementedError
