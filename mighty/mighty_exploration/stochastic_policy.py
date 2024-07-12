from __future__ import annotations

import torch
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class StochasticPolicy(MightyExplorationPolicy):
    """Entropy Based Exploration."""

    def __init__(self, algo, model, entropy_coefficient=0.2, discrete=True):
        """Initialize Entropy Based Exploration.

        :param algo: algorithm name
        :param model: policy model
        :param entropy_coefficient: entropy coefficient
        :return:
        """
        super().__init__(algo, model, discrete)
        self.entropy_coefficient = entropy_coefficient

        # FIXME: I did this already for the other exploration functions, but this would be nicer as a separate function
        def explore_func(s):
            state = torch.FloatTensor(s)  # Add batch dimension if needed

            if discrete:
                logits = self.model(state)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                weighted_log_prob = log_prob * entropy
            else:
                mean, std = self.model(state)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                weighted_log_prob = log_prob * entropy

            return action.detach().numpy(), weighted_log_prob.detach().numpy()

        self.explore_func = explore_func

    # FIXME: isn't this identical to the parent class? The only difference is the detach, no?
    def __call__(self, state, return_logp=False, metrics=None, evaluate=False):
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
            action, logprobs = self.sample_action(state)
            action = action.detach().numpy()
            output = (action, logprobs.detach.numpy()) if return_logp else action
        else:
            output = self.explore(state, return_logp, metrics)

        return output
