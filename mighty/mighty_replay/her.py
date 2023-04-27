import numpy as onp
import jax
import importlib
import gymnasium as gym
from coax.reward_tracing import TransitionBatch
from mighty.mighty_replay.mighty_replay_buffer import MightyReplay, flatten_infos


class HERGoalWrapper(gym.Wrapper):
    def __init__(self, env, goal, check_achieved):
        super().__init__(env)
        self.goal = goal
        f_name = check_achieved.split(".")[-1]
        import_from = importlib.import_module(".".join(check_achieved.split(".")[:-1]))
        self.check_goal_achieved = getattr(import_from, f_name)

    def reset(self):
        state, info = self.env.reset()
        info['g'] = self.goal
        info['ag'] = False
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        info['g'] = self.goal
        info['ag'] = self.check_goal_achieved(self.env, state, info, self.goal)
        return state, reward, terminated, truncated, info
    

class HER(MightyReplay):
    """

    Hindsight Experience Replay Buffer

    Parameters
    ----------
    capacity : positive int

        The capacity of the experience replay buffer.

    alpha : positive float, optional

        The sampling temperature :math:`\alpha>0`.

    beta : positive float, optional

        The importance-weight exponent :math:`\beta>0`.

    epsilon : positive float, optional

        The small regulator :math:`\epsilon>0`.

    random_seed : int, optional

        To get reproducible results.

    """

    def __init__(
        self,
        capacity,
        gamma,
        random_seed=None,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        reward_function=None,
        alternate_goal_function=None
    ):
        if not (isinstance(capacity, int) and capacity > 0):
            raise TypeError(f"capacity must be a positive int, got: {capacity}")

        self._capacity = int(capacity)
        self.goal_selection_strategy = goal_selection_strategy
        self.n_goals = n_sampled_goal
        self.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))
        self._random_seed = random_seed
        self._rnd = onp.random.RandomState(random_seed)
        self.clear()  # sets: self._deque, self._index
        self.contains_finished_episode = False
        self.gamma = gamma
        self.reward_func = reward_function
        if self.reward_func is None:
            def func(state, action, info):
                return int(info['ag'])
            self.reward_func = func
        else:
            f_name = self.reward_func.split(".")[-1]
            import_from = importlib.import_module(".".join(self.reward_func.split(".")[:-1]))
            self.reward_func = getattr(import_from, f_name)
        self.get_alternate_goal = alternate_goal_function
        assert self.get_alternate_goal is not None, "Function string to get an alternate goal has to be provided."
        f_name = self.get_alternate_goal.split(".")[-1]
        import_from = importlib.import_module(".".join(self.get_alternate_goal.split(".")[:-1]))
        self.get_alternate_goal = getattr(import_from, f_name)

    @property
    def capacity(self):
        return self._capacity

    def add(self, transition_batch, metrics):
        r"""

        Add a transition to the experience replay buffer.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        Adv : ndarray

            A batch of advantages, used to construct the priorities :math:`p_i`.

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
        if self.gamma in transition_batch.In:
            self.contains_finished_episode = True

    def sample(self, batch_size=32):
        r"""

        Get a batch of transitions to be used for bootstrapped updates.

        Parameters
        ----------
        batch_size : positive int, optional

            The desired batch size of the sample.

        Returns
        -------
        transitions : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

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
        real_batch = []
        for i in batch_indices:
            transition = self._storage[i].copy()
            # We need to transform this because jax can't handle dicts in the updates
            transition.extra_info = onp.array([list(flatten_infos(transition.extra_info))])
            real_batch.append(transition)
        return _concatenate_leaves(onp.array(real_batch))
    
    def _get_virtual_samples(self, batch_indices):
        virtual_batch = []
        for i in batch_indices:
            done_index = None
            idx = i
            # Find last state in same episode
            # If there is none, just return transition without altering
            while done_index is None and idx < self.capacity and self._storage[idx] is not None:
                if self._storage[idx].In == 0:
                    done_index = idx
                idx += 1

            if done_index is None:
                transition = self._storage[i].copy()
                # We need to transform this because jax can't handle dicts in the updates
                transition.extra_info = onp.array([list(flatten_infos(transition.extra_info))])
                virtual_batch.append(transition)
            else:
                transition = self._storage[i].copy()
                # If goal was reached: do nothing
                if not transition.extra_info['ag']:
                    # else set goal reached in info to true
                    transition.extra_info['ag'] = True
                    transition.extra_info['g'] = self.get_alternate_goal(transition.S, transition.extra_info, transition.extra_info['g'])
                    # adapt reward
                    reward = self.reward_func(transition.S, transition.A, transition.extra_info)
                else:
                    reward = transition.Rn[0]
                virtual_batch.append(
                    TransitionBatch.from_single(
                        transition.S[0],
                        transition.A[0],
                        transition.logP[0],
                        reward,
                        bool((transition.In==0)[0]),
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
        r"""Clear the experience replay buffer."""
        self._storage = onp.full(
            shape=(self.capacity,), fill_value=None, dtype="object")
        self._index = 0

    def __len__(self):
        return min(self.capacity, self._index)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._storage[: len(self)])


def _concatenate_leaves(pytrees):
    return jax.tree_map(lambda *leaves: onp.concatenate(leaves, axis=0), *pytrees)
