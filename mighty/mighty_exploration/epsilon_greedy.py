import jax
from jax import jit
import jax.numpy as jnp
import haiku as hk
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EpsilonGreedy(MightyExplorationPolicy):
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
        super().__init__(algo, func, env=None)
        self.epsilon = epsilon

        def func(params, state, rng, S, is_training):
            Q_s = self._Q_s(params, state, rng, S)

            A_greedy = (Q_s == Q_s.max(axis=1, keepdims=True)).astype(Q_s.dtype)
            A_greedy /= A_greedy.sum(
                axis=1, keepdims=True
            )  # there may be multiple max's (ties)
            A_greedy *= 1 - params["epsilon"]  # take away ε from greedy action(s)
            A_greedy += (
                params["epsilon"] / self.q.action_space.n
            )  # spread ε evenly to all actions

            dist_params = {"logits": jnp.log(A_greedy + 1e-15)}
            return dist_params, None  # return dummy function-state

        self._function = jit(func, static_argnums=(4,))

    @property
    def params(self):
        return hk.data_structures.to_immutable_dict(
            {"epsilon": self.epsilon, "q": self.q.params}
        )

    @params.setter
    def params(self, new_params):
        if jax.tree_util.tree_structure(new_params) != jax.tree_util.tree_structure(
            self.params
        ):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params["epsilon"]
        self.q.params = new_params["q"]

    @property
    def function(self):
        return self._function

    @property
    def function_state(self):
        return self.q.function_state

    @property
    def rng(self):
        return self.q.rng

    @function_state.setter
    def function_state(self, new_function_state):
        self.q.function_state = new_function_state
