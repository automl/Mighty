import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import haiku as hk
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EZGreedy(MightyExplorationPolicy):
    """EZGreedy exploration."""

    def __init__(
        self,
        algo,
        func,
        epsilon=0.1,
        zipf_param=2,
        env=None,
        observation_preprocessor=None,
        proba_dist=None,
        random_seed=None,
    ):
        """
        Initialize EZGreedy.

        :param algo: algorithm name
        :param func: policy function
        :param epsilon: exploration epsilon
        :param zipf_param: parameter for zipf action length distribution 
        :param env: environment
        :param observation_preprocessor: preprocessing for observation
        :param proba_dist: probability distribution
        :param random_seed: seed for sampling
        :return:
        """

        self.epsilon = epsilon
        self.zipf_param = zipf_param
        self.skip = max(1, np.random.default_rng().zipf(self.zipf_param))
        self.skip = max(1, np.random.default_rng().zipf(2))
        self.skipped = 0
        self.action = None

        super().__init__(
            algo,
            func,
            env=env,
            observation_preprocessor=observation_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed,
        )

        def func(params, state, rng, S, is_training):
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

    def explore(self, s, return_logp=False, metrics=None):
        """
        Explore.

        :param s: state
        :param return_logp: return logprobs
        :param metrics: not used
        :return: action or (action, logprobs)
        """

        if self.skipped >= self.skip or self.action is None:
            self.action, self.logprobs = self.sample_action(s)
            self.skip = max(1, np.random.default_rng().zipf(2))
            self.skipped = 0
        else:
            self.skipped += 1
        return (self.action, self.logprobs) if return_logp else self.action

    @property
    def params(self):
        """Get params."""
        return hk.data_structures.to_immutable_dict(
            {"epsilon": self.epsilon, "q": self.q.params, "skip": self.skip}
        )

    @params.setter
    def params(self, new_params):
        """Set params."""
        if jax.tree_util.tree_structure(new_params) != jax.tree_util.tree_structure(
            self.params
        ):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params["epsilon"]
        self.skip = new_params["skip"]
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
