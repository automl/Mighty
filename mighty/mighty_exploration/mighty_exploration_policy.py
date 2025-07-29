"""Mighty Exploration Policy."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Normal


def sample_nondeterministic_logprobs(
    z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, keepdim: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.exp(log_std)  # [batch, action_dim]
    dist = Normal(mean, std)

    # 2a) log_pz = ∑ᵢ log N(zᵢ; μᵢ, σᵢ)
    log_pz = dist.log_prob(z).sum(dim=-1, keepdim=keepdim)  # [batch]

    # 2b) tanh‐correction = ∑ᵢ log(1 − tanh(zᵢ)² + ε)
    eps = 1e-6
    log_correction = torch.log(1.0 - torch.tanh(z).pow(2) + eps).sum(dim=-1, keepdim=keepdim)  # [batch]

    # 2c) final log_prob of a = tanh(z)
    log_prob = log_pz - log_correction  # [batch]
    return log_prob


class MightyExplorationPolicy:
    """Generic Exploration Policy Interface.

    Now supports:
      - Discrete: `model(state)` → logits → Categorical
      - Continuous (squashed-Gaussian): `model(state)` → (action, z, mean, log_std)
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

        # ─── Continuous squashed‐Gaussian (4‐tuple) ──────────────────────────
        out = self.model(state)
        if isinstance(out, tuple) and len(out) == 4:
            action = out[0]  # [batch, action_dim]
            log_prob = sample_nondeterministic_logprobs(z=out[1], mean=out[2], log_std=out[3], keepdim=self.algo=="sac")
            return action.detach().cpu().numpy(), log_prob

        # ─── Fallback: if model(state) returns a Distribution ────────────────
        elif isinstance(out, torch.distributions.Distribution):
            dist = out  # user returned a Distribution
            action = dist.sample()  # [batch]
            log_prob = dist.log_prob(action)  # [batch]
            return action.detach().cpu().numpy(), log_prob

        # ─── Otherwise, we don’t know how to sample ─────────────────────────
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
