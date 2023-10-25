import importlib

import gymnasium as gym
import jax
import numpy as onp
from coax.reward_tracing import TransitionBatch

from mighty.mighty_replay.mighty_replay_buffer import MightyReplay, flatten_infos


class HERGoalWrapper(gym.Wrapper):
    """Goal Visibility Wrapper for HER."""

    def __init__(self, env, goal, check_achieved):
        """
        Initialize Wrapper.

        :param env: environment
        :param goal: goal
        :param check_achieved: function name for checking if goal is achieved
        :return:
        """

        super().__init__(env)
        self.goal = goal
        f_name = check_achieved.split(".")[-1]
        import_from = importlib.import_module(".".join(check_achieved.split(".")[:-1]))
        self.check_goal_achieved = getattr(import_from, f_name)

    def reset(self):
        """
        Reset environment.

        :return: state, info
        """

        state, info = self.env.reset()
        info["g"] = self.goal
        info["ag"] = False
        return state, info

    def step(self, action):
        """
        Environment Step.

        :param action: action
        :return: state, reward, terminated, truncated, info
        """

        state, reward, terminated, truncated, info = self.env.step(action)
        info["g"] = self.goal
        info["ag"] = self.check_goal_achieved(self.env, state, info, self.goal)
        return state, reward, terminated, truncated, info


class HER(MightyReplay):
    """Hindsight Experience Replay Buffer."""

    def __init__(
        self,
        capacity,
        gamma,
        random_seed=None,
        her_ratio: int = 4,
        reward_function=None,
        alternate_goal_function=None,
    ):
        """
        Init HER.

        :param capacity: buffer size
        :param gamma: discount factor
        :param random_seed: seed for sampling
        :param her_ratio: ratio of sampled to real goals
        :param reward_function: string name (package.module.func_name) of reward function
        :param alternate_goal_function: function for computing alternate goals
        :return:
        """

        if not (isinstance(capacity, int) and capacity > 0):
            raise TypeError(f"capacity must be a positive int, got: {capacity}")

        self._capacity = int(capacity)
        self.her_ratio = her_ratio
        self._random_seed = random_seed
        self._rnd = onp.random.RandomState(random_seed)
        self.clear()  # sets: self._deque, self._index
        self.contains_finished_episode = False
        self.gamma = gamma
        self.reward_func = reward_function
        if self.reward_func is None:

            def func(state, action, info):
                return int(info["ag"])

            self.reward_func = func
        else:
            f_name = self.reward_func.split(".")[-1]
            import_from = importlib.import_module(
                ".".join(self.reward_func.split(".")[:-1])
            )
            self.reward_func = getattr(import_from, f_name)
        self.get_alternate_goal = alternate_goal_function
        assert (
            self.get_alternate_goal is not None
        ), "Function string to get an alternate goal has to be provided."
        f_name = self.get_alternate_goal.split(".")[-1]
        import_from = importlib.import_module(
            ".".join(self.get_alternate_goal.split(".")[:-1])
        )
        self.get_alternate_goal = getattr(import_from, f_name)

    @property
    def capacity(self):
        """Maximum size."""
        return self._capacity

    def add(self, transition_batch, metrics):
        """
        Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """

        # Store Transition
        if not isinstance(transition_batch, TransitionBatch):
            raise TypeError(
                f"transition_batch must be a TransitionBatch, got: {type(transition_batch)}"
            )

        transition_batch.idx = self._index + onp.arange(transition_batch.batch_size)
        idx = transition_batch.idx % self.capacity  # wrap around
        self._storage[idx] = list(transition_batch.to_singles())
        self._index += transition_batch.batch_size
        if 0 in transition_batch.In and not self.contains_finished_episode:
            self.contains_finished_episode = True

    def sample(self, batch_size=32):
        """
        Sample batch.

        :param batch_size: batch size
        :return: batch
        """

        if not onp.any(self.contains_finished_episode):
            raise RuntimeError(
                "No episode has finished at this point. Please consider using a smaller number of episode steps or a larger value for learning starts."
            )

        idx = onp.random.choice(onp.arange(self._index), size=batch_size)
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_indices, real_batch_indices = onp.split(idx, [nb_virtual])

        # Create virtual transitions by sampling new desired goals and computing new reward
        virtual_data = self._get_virtual_samples(virtual_batch_indices)
        real_data = self.get_real_batch(real_batch_indices)

        transition_batch = _concatenate_leaves([real_data, virtual_data])
        return transition_batch

    def get_real_batch(self, batch_indices):
        """
        Get actual transitions.

        :param batch_indices: indices to get
        :return: batch
        """

        real_batch = []
        for i in batch_indices:
            transition = self._storage[i].copy()
            # We need to transform this because jax can't handle dicts in the updates
            transition.extra_info = onp.array(
                [list(flatten_infos(transition.extra_info))]
            )
            real_batch.append(transition)
        return _concatenate_leaves(onp.array(real_batch))

    def _get_virtual_samples(self, batch_indices):
        """
        Get virtual batch.

        :param batch_indices: indices to get
        :return: virtual batch
        """
        virtual_batch = []
        for i in batch_indices:
            done_index = None
            idx = i
            # Find last state in same episode
            # If there is none, just return transition without altering
            while (
                done_index is None
                and idx < self.capacity
                and self._storage[idx] is not None
            ):
                if self._storage[idx].In == 0:
                    done_index = idx
                idx += 1

            if done_index is None:
                transition = self._storage[i].copy()
                # We need to transform this because jax can't handle dicts in the updates
                transition.extra_info = onp.array(
                    [list(flatten_infos(transition.extra_info))]
                )
                virtual_batch.append(transition)
            else:
                transition = self._storage[i].copy()
                # If goal was reached: do nothing
                if not transition.extra_info["ag"]:
                    # else set goal reached in info to true
                    transition.extra_info["ag"] = True
                    transition.extra_info["g"] = self.get_alternate_goal(
                        transition.S, transition.extra_info, transition.extra_info["g"]
                    )
                    # adapt reward
                    reward = self.reward_func(
                        transition.S, transition.A, transition.extra_info
                    )
                else:
                    reward = transition.Rn[0]
                virtual_batch.append(
                    TransitionBatch.from_single(
                        transition.S[0],
                        transition.A[0],
                        transition.logP[0],
                        reward,
                        bool((transition.In == 0)[0]),
                        self.gamma,
                        transition.S_next[0],
                        transition.A_next[0],
                        transition.logP_next[0],
                        transition.W[0],
                        transition.idx[0],
                        onp.array(list(flatten_infos(transition.extra_info))),
                    )
                )
        return _concatenate_leaves(onp.array(virtual_batch))

    def clear(self):
        """Clear buffer."""
        self._storage = onp.full(
            shape=(self.capacity,), fill_value=None, dtype="object"
        )
        self._index = 0

    def __len__(self):
        """Return current size."""
        return min(self.capacity, self._index)

    def __bool__(self):
        """Return not empty."""
        return bool(len(self))

    def __iter__(self):
        """Get iterator."""
        return iter(self._storage[: len(self)])


def _concatenate_leaves(pytrees):
    return jax.tree_map(lambda *leaves: onp.concatenate(leaves, axis=0), *pytrees)
