"""Epsilon Greedy Exploration."""

from __future__ import annotations

import numpy as np
from mighty.mighty_exploration import EpsilonGreedy


class EZGreedy(EpsilonGreedy):
    """Epsilon Greedy Exploration."""

    def __init__(
        self,
        algo,
        model,
        epsilon=0.1,
        zipf_param=2,
    ):
        """Initialize EZ Greedy.

        :param algo: algorithm name
        :param model: model
        :param epsilon: exploration epsilon
        :param zipf_param: parametrizes the Zipf distribution for skipping
        :return:
        """
        super().__init__(algo, model)
        self.epsilon = epsilon
        self.zipf_param = zipf_param
        self.skip = max(1, np.random.default_rng().zipf(self.zipf_param))
        self.skipped = None
        self.frozen_actions = None

    def explore_func(self, s):
        # Epsilon Greedy Step
        greedy_actions, qvals = self.sample_action(s)

        # Initialize Skips
        if self.skipped is None:
            self.skipped = np.zeros(len(greedy_actions))
            self.frozen_actions = np.zeros(greedy_actions.shape)

        # Do epsilon greedy exploration
        exploration_flags, random_actions = self.get_random_actions(
            len(greedy_actions), len(qvals[0])
        )
        actions = np.where(exploration_flags, random_actions, greedy_actions)

        # Decay Skips
        self.skipped = np.maximum(0, self.skipped - 1)

        # Sample skip lengths for new exploration steps
        new_skips = np.where(
            exploration_flags,
            [self.rng.zipf(self.zipf_param) for _ in range(len(exploration_flags))],
            [0] * len(exploration_flags),
        )
        for i in range(len(self.skipped)):
            if self.skipped[i] == 0:
                self.frozen_actions[i] = actions[i]

            if exploration_flags[i] and self.skipped[i] == 0:
                self.skipped[i] = new_skips[i]

        # Apply skip
        skips = [self.skipped[i] > 0 for i in range(len(self.skipped))]
        actions = np.where(skips, self.frozen_actions, actions)
        return actions.astype(int), qvals
