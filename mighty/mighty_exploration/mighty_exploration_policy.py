from coax import Policy
from coax._core.value_based_policy import BaseValueBasedPolicy
from coax.utils import batch_to_single


class MightyExplorationPolicy(Policy, BaseValueBasedPolicy):
    """Generic Policy."""

    def __init__(
        self,
        algo,
        func,
        env=None,
        observation_preprocessor=None,
        proba_dist=None,
        random_seed=None,
    ) -> None:
        """
        Initialize Exploration Strategy.

        :param algo: algorithm name
        :param func: policy function
        :param env: environment
        :param observation_preprocessor: preprocessing for observation
        :param proba_dist: probability distribution
        :param random_seed: seed for sampling
        :return:
        """

        self.algo = algo
        if algo == "q":
            BaseValueBasedPolicy.__init__(self, func)
        else:
            assert (
                env is not None
            ), "Environment must be given in policy exploration methods."
            Policy.__init__(
                self,
                func=func,
                env=env,
                observation_preprocessor=observation_preprocessor,
                proba_dist=proba_dist,
                random_seed=random_seed,
            )

    # This is the original call in coax
    def sample_action(self, s):
        """
        Sample from policy.

        :param s: state
        :return: (action, logprobs)
        """

        S = self.observation_preprocessor(self.rng, s)
        X, logP = self.sample_func(self.params, self.function_state, self.rng, S)
        x = self.proba_dist.postprocess_variate(self.rng, X)
        return (x, batch_to_single(logP))

    def __call__(self, s, return_logp=False, metrics={}, eval=False):
        """
        Get action.

        :param s: state
        :param return_logp: return logprobs
        :param metrics: current metric dict
        :param eval: eval mode
        :return: action or (action, logprobs)
        """

        if eval:
            action = self.mode(s)
            return (action, []) if return_logp else action
        else:
            return self.explore(s, return_logp, metrics)

    def explore(self, s, return_logp, _):
        """
        Explore.

        :param s: state
        :param return_logp: return logprobs
        :param _: not used
        :return: action or (action, logprobs)
        """

        action, logprobs = self.sample_action(s)
        return (action, logprobs) if return_logp else action
