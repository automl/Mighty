import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import haiku as hk
from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class EZGreedy(MightyExplorationPolicy):
    r"""
    Create an :math:`\epsilon`z-greedy policy, given a q-function.
    Works like a normal epsilon-greedy function but repeats the sampled action.

    Parameters
    ----------
    q : Q
        A state-action value function.
    epsilon : float between 0 and 1, optional
        The probability of sampling an action uniformly at random (as opposed to sampling greedily).
    skip : int
        Number of steps to repeat the sampled action
    """

    def __init__(
        self,
        algo,
        func,
        epsilon=0.1,
        skip=100,
        env=None,
        observation_preprocessor=None,
        proba_dist=None,
        random_seed=None,
    ):
        self.epsilon = epsilon
        self.skip = skip
        self.skipped = 0
        self.current_params = None

        super().__init__(
            algo,
            func,
            env=env,
            observation_preprocessor=observation_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed,
        )

        def func(params, state, rng, S, is_training):
            self.skipped = 0
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
            self.current_params = dist_params
            return dist_params, None  # return dummy function-state

        self._function = jit(func, static_argnums=(4,))

    def __call__(self, s, return_logp=False, metrics=None, eval=False):
        if self.skipped >= self.skip or self.current_params is None:
            self.action, self.logprobs = self.sample_action(s)
            self.skip = np.random.default_rng().zipf(2)
            self.skipped = 0
        else:
            self.skipped += 1
        return (self.action, self.logprobs) if return_logp else self.action

    @property
    def params(self):
        return hk.data_structures.to_immutable_dict(
            {"epsilon": self.epsilon, "q": self.q.params, "skip": self.skip}
        )

    @params.setter
    def params(self, new_params):
        if jax.tree_util.tree_structure(new_params) != jax.tree_util.tree_structure(
            self.params
        ):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params["epsilon"]
        self.skip = new_params["skip"]
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
