"""Stochastic Policy for Entropy-Based Exploration."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from mighty.mighty_exploration.mighty_exploration_policy import (
    MightyExplorationPolicy,
    sample_nondeterministic_logprobs,
)
from mighty.mighty_models import SACModel


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

        self.model = model

        super().__init__(algo, model, discrete)
        self.entropy_coefficient = entropy_coefficient
        self.discrete = discrete

        # --- override sample_action only for continuous SAC ---
        if not discrete and isinstance(model, SACModel):
            # for evaluation use deterministic=True; training will go through .explore()
            def _sac_sample(state_np):
                state = torch.as_tensor(state_np, dtype=torch.float32)
                # forward returns (action, z, mean, log_std)
                action, z, mean, log_std = model(state, deterministic=True)
                logp = model.policy_log_prob(z, mean, log_std)
                return action.detach().cpu().numpy(), logp

            self.sample_action = _sac_sample

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
            # Get model output
            model_output = self.model(state)

            # Handle different model output formats

            # NEW: 3-tuple case (Standard PPO): (action, mean, log_std)
            if isinstance(model_output, tuple) and len(model_output) == 3:
                action, mean, log_std = model_output
                std = torch.exp(log_std)
                dist = Normal(mean, std)

                # Direct log prob (no tanh correction)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

                if return_logp:
                    return action.detach().cpu().numpy(), log_prob
                else:
                    weighted_log_prob = log_prob * self.entropy_coefficient
                    return action.detach().cpu().numpy(), weighted_log_prob

            # 4-tuple case (Tanh squashing): (action, z, mean, log_std)
            elif isinstance(model_output, tuple) and len(model_output) == 4:
                action, z, mean, log_std = model_output
                log_prob = sample_nondeterministic_logprobs(
                    z=z, mean=mean, log_std=log_std,
                    tanh_squash=getattr(self.model, "tanh_squash", False),
                )

                if return_logp:
                    return action.detach().cpu().numpy(), log_prob
                else:
                    weighted_log_prob = log_prob * self.entropy_coefficient
                    return action.detach().cpu().numpy(), weighted_log_prob

            # Check for model attribute-based approaches
            elif hasattr(self.model, "continuous_action") and getattr(
                self.model, "continuous_action"
            ):
                # This handles the case where model has continuous_action attribute
                # but we need to determine the output format dynamically
                if len(model_output) == 3:
                    # Standard PPO mode: (action, mean, log_std)
                    action, mean, log_std = model_output
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                elif len(model_output) == 4:
                    # Tanh squashing mode: (action, z, mean, log_std)
                    action, z, mean, log_std = model_output
                    log_prob = sample_nondeterministic_logprobs(
                        z=z, mean=mean, log_std=log_std,
                        tanh_squash=getattr(self.model, "tanh_squash", False),
                    )
                else:
                    raise ValueError(
                        f"Unexpected model output length: {len(model_output)}"
                    )

                if return_logp:
                    return action.detach().cpu().numpy(), log_prob
                else:
                    weighted_log_prob = log_prob * self.entropy_coefficient
                    return action.detach().cpu().numpy(), weighted_log_prob

            # Check for output_style attribute (backwards compatibility)
            elif hasattr(self.model, "output_style"):
                if self.model.output_style == "squashed_gaussian":
                    # Should be 4-tuple: (action, z, mean, log_std)
                    action, z, mean, log_std = model_output
                    log_prob = sample_nondeterministic_logprobs(
                        z=z, mean=mean, log_std=log_std,
                        tanh_squash=getattr(self.model, "tanh_squash", False),
                    )

                    if return_logp:
                        return action.detach().cpu().numpy(), log_prob
                    else:
                        weighted_log_prob = log_prob * self.entropy_coefficient
                        return action.detach().cpu().numpy(), weighted_log_prob

                elif self.model.output_style == "mean_std":
                    # Should be 2-tuple: (mean, std)
                    mean, std = model_output
                    dist = Normal(mean, std)
                    z = dist.rsample()
                    action = torch.tanh(z)
                    log_std = torch.log(std)
                    log_prob = sample_nondeterministic_logprobs(
                        z=z, mean=mean, log_std=log_std,
                        tanh_squash=getattr(self.model, "tanh_squash", False),
                    )

                    entropy = dist.entropy().sum(dim=-1, keepdim=True)
                    weighted_log_prob = log_prob * entropy
                    return action.detach().cpu().numpy(), weighted_log_prob

                else:
                    raise RuntimeError(
                        f"StochasticPolicy: unknown output_style '{self.model.output_style}'"
                    )

            # Special handling for SACModel
            elif self.algo == "sac" and isinstance(self.model, SACModel):
                action, z, mean, log_std = self.model(state, deterministic=False)
                # CRITICAL: Use the model's policy_log_prob which includes tanh correction
                log_prob = self.model.policy_log_prob(z, mean, log_std)
                return action.detach().cpu().numpy(), log_prob

            else:
                raise RuntimeError(
                    "StochasticPolicy: cannot interpret model(state) output of type "
                    f"{type(model_output)} with length {len(model_output) if isinstance(model_output, tuple) else 'N/A'}"
                )

    def forward(self, s):
        """
        Alias for explore, so policy(s) returns (action, weighted_log_prob).
        """
        return self.explore(s, return_logp=False)
