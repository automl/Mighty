""""Stochastic Policy for Entropy-Based Exploration."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from mighty.mighty_exploration.mighty_exploration_policy import (
    MightyExplorationPolicy,
    sample_nondeterministic_logprobs,
)


class StochasticPolicy(MightyExplorationPolicy):
    """Entropy-Based Exploration for discrete and continuous action spaces."""

    def __init__(
        self, algo, model, entropy_coefficient: float = 0.2, discrete: bool = True
    ):
        """
        :param algo: the RL algorithm instance
        :param model: the policy model
        :param entropy_coefficient: weight on entropy term
        :param discrete: whether the action space is discrete
        """
        super().__init__(algo, model, discrete)
        self.entropy_coefficient = entropy_coefficient
        self.discrete = discrete

    def explore(self, s, return_logp, metrics=None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Given observations `s`, sample an exploratory action and compute a weighted log-prob.

        Returns:
          action: numpy array of actions
          weighted_log_prob: Tensor of shape [batch, 1]
        """
        state = torch.as_tensor(s, dtype=torch.float32)
        if self.discrete:
            logits = self.model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
            return action.detach().cpu().numpy(), log_prob * self.entropy_coefficient
        else:
            # If model has attribute continuous_action=True, we know:
            #   model(state) → (action, z, mean, log_std)
            if self.model.output_style == "squashed_gaussian":
                # 1) Forward pass: get (action, z, mean, log_std)
                action, z, mean, log_std = self.model(
                    state
                )  # each: [batch, action_dim]
                log_prob = sample_nondeterministic_logprobs(
                    action=action,
                    z=z,
                    mean=mean,
                    log_std=log_std,
                    sac=self.algo == "sac",
                )
                if return_logp:
                    return action.detach().cpu().numpy(), log_prob
                else:
                    weighted_log_prob = log_prob * self.entropy_coefficient
                    return action.detach().cpu().numpy(), weighted_log_prob
            # If it’s “mean, std”‐style continuous (rare in our code), handle that case
            elif self.model.output_style == "mean_std":
                mean, std = self.model(state)
                dist = Normal(mean, std)
                z = dist.rsample()  # [batch, action_dim]
                action = torch.tanh(z)  # [batch, action_dim]

                log_prob = sample_nondeterministic_logprobs(
                    z=z, mean=mean, log_std=torch.log(std), sac=self.algo == "sac"
                )
                entropy = dist.entropy().sum(dim=-1, keepdim=True)  # [batch, 1]
                weighted_log_prob = log_prob * entropy
                return action.detach().cpu().numpy(), weighted_log_prob
            else:
                raise RuntimeError(
                    "StochasticPolicy: cannot interpret model(state) output of type "
                    f"{type(self.model(state))}"
                )

    def forward(self, s):
        """
        Alias for explore, so policy(s) returns (action, weighted_log_prob).
        """
        return self.explore(s)
