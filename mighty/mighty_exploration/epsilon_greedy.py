import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit

from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EpsilonGreedy(MightyExplorationPolicy):
    """Epsilon Greedy Exploration."""

    def __init__(
        self,
        algo,
        func,
        epsilon=0.1,
        env=None,
        observation_preprocessor=None,
        proba_dist=None,
        random_seed=None,
    ):
        """
        Initialize Epsilon Greedy.

        :param algo: algorithm name
        :param func: policy function
        :param epsilon: exploration epsilon
        :param env: environment
        :param observation_preprocessor: preprocessing for observation
        :param proba_dist: probability distribution
        :param random_seed: seed for sampling
        :return:
        """

        super().__init__(algo, func, env=None)
        self.epsilon = epsilon

        def func(params, state, rng, S, is_training):
            """Note: is_training actually means 'is_eval' here due to coax."""
            Q_s = self._Q_s(params, state, rng, S)

            A_greedy = (Q_s == Q_s.max(axis=1, keepdims=True)).astype(Q_s.dtype)
            A_greedy /= A_greedy.sum(
                axis=1, keepdims=True
            )  # there may be multiple max's (ties)

            if not is_training:
                A_greedy *= 1 - params["epsilon"]  # take away ε from greedy action(s)
                A_greedy += (
                    params["epsilon"] / self.q.action_space.n
                )  # spread ε evenly to all actions

            dist_params = {"logits": jnp.log(A_greedy + 1e-15)}
            return dist_params, None  # return dummy function-state

        self._function = jit(func, static_argnums=(4,))

    @property
    def params(self):
        """Get params."""
        return hk.data_structures.to_immutable_dict(
            {"epsilon": self.epsilon, "q": self.q.params}
        )

    @params.setter
    def params(self, new_params):
        """Set params."""
        if jax.tree_util.tree_structure(new_params) != jax.tree_util.tree_structure(
            self.params
        ):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params["epsilon"]
        self.q.params = new_params["q"]

    @property
    def function(self):
        """Get function."""
        return self._function

    @property
    def function_state(self):
        """Get function state."""
        return self.q.function_state

    @property
    def rng(self):
        """Get RNG."""
        return self.q.rng

    @function_state.setter
    def function_state(self, new_function_state):
        """Set function state."""
        self.q.function_state = new_function_state
