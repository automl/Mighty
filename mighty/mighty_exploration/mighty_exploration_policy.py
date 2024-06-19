"""Mighty Exploration Policy."""

from __future__ import annotations

import numpy as np
import torch


class MightyExplorationPolicy:
    """Generic Policy."""

    def __init__(
        self,
        algo,
        model,
        discrete=False,
    ) -> None:
        """Initialize Exploration Strategy.

        :param algo: algorithm name
        :param func: policy function
        :return:
        """
        self.rng = np.random.default_rng()
        self.algo = algo
        self.model = model

        # Undistorted action sampling
        if self.algo == "q":

            def sample_func(s):
                s = torch.as_tensor(s, dtype=torch.float32)
                qs = self.model(s)
                return np.argmax(qs.detach(), axis=1), qs

        else:

            def sample_func(s, std=None):
                pred = self.model(s)
                if discrete:
                    dist = torch.distributions.Categorical(pred)
                else:
                    dist = torch.distributions.MultivariateNormal(pred, std)
                action = dist.sample()
                return action.detach(), dist.log_prob.detach()

        self.sample_action = sample_func

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
            action = action.detach().numpy()
            output = (action, logprobs) if return_logp else action
        else:
            output = self.explore(s, return_logp, metrics)
        return output

    def explore(self, s, return_logp, _):
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
