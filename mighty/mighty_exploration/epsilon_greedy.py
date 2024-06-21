"""Epsilon Greedy Exploration."""

from __future__ import annotations

import numpy as np
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EpsilonGreedy(MightyExplorationPolicy):
    """Epsilon Greedy Exploration."""

    def __init__(
        self,
        algo,
        model,
        epsilon=0.1,
    ):
        """Initialize Epsilon Greedy.

        :param algo: algorithm name
        :param func: policy function
        :param epsilon: exploration epsilon
        :param env: environment
        :return:
        """
        super().__init__(algo, model)
        self.epsilon = epsilon

        def explore_func(s):
            greedy_actions, qvals = self.sample_action(s)
            if isinstance(epsilon, float):
                exploration_flags = [
                    self.rng.random() < self.epsilon for _ in range(len(greedy_actions))
                ]
            else:
                index = 0
                exploration_flags = []
                while len(exploration_flags) < len(greedy_actions):
                    exploration_flags.append(self.rng.random() < self.epsilon[index])
                    index += 1
                    if index >= len(self.epsilon):
                        index = 0

            exploration_flags = np.array(exploration_flags)
            random_actions = self.rng.integers(len(qvals[0]), size=len(greedy_actions))
            actions = np.where(exploration_flags, random_actions, greedy_actions)
            return actions.astype(int), qvals

        self.explore_func = explore_func
