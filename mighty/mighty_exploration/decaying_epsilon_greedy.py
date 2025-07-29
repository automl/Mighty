"""Decaying Epsilon‐Greedy Exploration."""

from __future__ import annotations

import numpy as np

from mighty.mighty_exploration.epsilon_greedy import EpsilonGreedy


class DecayingEpsilonGreedy(EpsilonGreedy):
    """Epsilon-Greedy Exploration with linear decay schedule."""

    def __init__(
        self,
        algo,
        model,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay_steps: int = 10000,
    ):
        """
        :param algo:       algorithm name
        :param model:      policy model (e.g. Q-network)
        :param epsilon_start: Initial ε (at time step 0)
        :param epsilon_final: Final ε (after decay_steps)
        :param epsilon_decay_steps: Number of steps over which to linearly
                                     decay ε from epsilon_start → epsilon_final.
        """
        super().__init__(algo=algo, model=model, epsilon=epsilon_start)
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def _compute_epsilon(self) -> float:
        """Linearly interpolate between epsilon_start and epsilon_final."""
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_final
        fraction = self.total_steps / self.epsilon_decay_steps
        return float(
            self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)
        )

    def get_random_actions(self, n_actions, action_length):
        """
        Override to recompute ε at each call, then delegate to EpsilonGreedy's logic.
        """
        # 1) Update ε based on total_steps
        current_epsilon = self._compute_epsilon()
        self.epsilon = current_epsilon

        # 2) Call parent method to build exploration flags & random actions
        exploration_flags, random_actions = super().get_random_actions(
            n_actions, action_length
        )

        # 3) Advance the step counter (so subsequent calls see a smaller ε)
        self.total_steps += n_actions

        return exploration_flags, random_actions

    def explore_func(self, s):
        """Same as EpsilonGreedy, except uses decayed ε each time."""
        greedy_actions, qvals = self.sample_action(s)
        exploration_flags, random_actions = self.get_random_actions(
            len(greedy_actions), len(qvals[0])
        )
        actions = np.where(exploration_flags, random_actions, greedy_actions)
        return actions.astype(int), qvals
