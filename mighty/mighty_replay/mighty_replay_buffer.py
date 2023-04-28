from coax.experience_replay._simple import SimpleReplayBuffer
from coax.experience_replay._prioritized import PrioritizedReplayBuffer
from collections.abc import Iterable


def flatten_infos(xs):
    """
    Transform info dict to flat list.

    :param xs: info dict
    :return: flattened infos
    """

    if isinstance(xs, dict):
        xs = list(xs.values())
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_infos(x)
        else:
            yield x


class MightyReplay(SimpleReplayBuffer):
    """Simple replay buffer."""

    def __init__(
        self, capacity, random_seed=None, keep_infos=False, flatten_infos=False
    ):
        """
        Initialize Buffer.

        :param capacity: Buffer size
        :param random_seed: Seed for sampling
        :param keep_infos: Keep the extra info dict. Required for some algorithms.
        :param flatten_infos: Make flat list from infos. Might be necessary, depending on info content.
        :return:
        """

        super().__init__(capacity, random_seed)
        self.keep_infos = keep_infos
        self.flatten_infos = flatten_infos

    def add(self, transition_batch, metrics):
        """
        Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """

        if not self.keep_infos:
            transition_batch.extra_info = []
        elif self.flatten_infos:
            transition_batch.extra_info = [
                list(flatten_infos(transition_batch.extra_info))
            ]
        super().add(transition_batch)


class PrioritizedReplay(PrioritizedReplayBuffer):
    """Prioritized Replay Buffer."""

    def __init__(
        self,
        capacity,
        alpha=1.0,
        beta=1.0,
        epsilon=1e-4,
        random_seed=None,
        keep_infos=False,
        flatten_infos=False,
    ):
        """
        Initialize Buffer.

        :param capacity: Buffer size
        :param alpha: Priorization exponent
        :param beta: Bias exponent
        :param epsilon: Step size
        :param random_seed: Seed for sampling
        :param keep_infos: Keep the extra info dict. Required for some algorithms.
        :param flatten_infos: Make flat list from infos. Might be necessary, depending on info content.
        :return:
        """

        super().__init__(capacity, alpha, beta, epsilon, random_seed)
        self.keep_infos = keep_infos
        self.flatten_infos = flatten_infos

    def add(self, transition_batch, metrics):
        """
        Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """

        if not self.keep_infos:
            transition_batch.extra_info = []
        elif self.flatten_infos:
            transition_batch.extra_info = [
                list(flatten_infos(transition_batch.extra_info))
            ]

        advantage = metrics["td_error"]
        super().add(transition_batch, advantage)
