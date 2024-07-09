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

            def sample_func(state):
                state = torch.as_tensor(state, dtype=torch.float32)
                qs = self.model(state)
                return np.argmax(qs.detach(), axis=1), qs

        else:

            def sample_func(state):
                state = torch.FloatTensor(state)

                if discrete:
                    pred = self.model(state)
                    dist = torch.distributions.Categorical(logits=pred)
                else:
                    pred, std = self.model(state)
                    dist = torch.distributions.Normal(pred, std)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action, log_prob

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
