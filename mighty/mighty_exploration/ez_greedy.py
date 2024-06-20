"""Epsilon Greedy Exploration."""

from __future__ import annotations

import numpy as np
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EZGreedy(MightyExplorationPolicy):
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

        def explore_func(s):
            # Epsilon Greedy Step
            greedy_actions, qvals = self.sample_action(s)
            exploration_flag_dim_1 = qvals.shape[-1] if len(qvals.shape) > 1 else 1

            if self.skipped is None:
                self.skipped = np.zeros(len(greedy_actions))
                self.frozen_actions = np.zeros(greedy_actions.shape)

            if isinstance(epsilon, float):
                exploration_flags = [
                    [self.rng.random() < self.epsilon] * exploration_flag_dim_1
                    for _ in range(len(greedy_actions))
                ]
            else:
                index = 0
                exploration_flags = []
                while len(exploration_flags) < len(greedy_actions):
                    exploration_flags.append(
                        [self.rng.random() < self.epsilon[index]]
                        * exploration_flag_dim_1
                    )
                    index += 1
                    if index >= len(self.epsilon):
                        index = 0

            exploration_flags = np.array(exploration_flags)
            random_actions = self.rng.integers(len(qvals[0]), size=greedy_actions.shape)
            actions = np.where(
                exploration_flags.squeeze(), random_actions, greedy_actions
            )

            # Decay Skips
            self.skipped = np.maximum(0, self.skipped - 1)

            # Sample skip lengths for new exploration steps
            new_skips = np.where(
                exploration_flags[:, 0],
                [self.rng.zipf(self.zipf_param) for _ in range(len(exploration_flags))],
                [0] * len(exploration_flags),
            )
            for i in range(len(self.skipped)):
                if exploration_flags[i][0] and self.skipped[i] == 0:
                    self.skipped[i] = new_skips[i]
                    self.frozen_actions[i] = actions[i]

            # Apply skip
            skips = [
                [self.skipped[i] > 0] * qvals.shape[-1]
                for i in range(len(self.skipped))
            ]
            actions = np.where(skips, self.frozen_actions, actions)
            return actions, qvals

        self.explore_func = explore_func
