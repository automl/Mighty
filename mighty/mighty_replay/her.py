import numpy as onp
import chex
import jax
from coax.reward_tracing import TransitionBatch
from mighty.mighty_replay import MightyReplay

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
    def __init__(self, capacity, gamma, random_seed=None, n_sampled_goal: int = 4, goal_selection_strategy: str = "future", reward_function=None):
        if not (isinstance(capacity, int) and capacity > 0):
            raise TypeError(f"capacity must be a positive int, got: {capacity}")

        self._capacity = int(capacity)
        self.goal_selection_strategy = goal_selection_strategy
        self.n_goals = n_sampled_goal
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        self._random_seed = random_seed
        self._rnd = onp.random.RandomState(random_seed)
        self.clear()  # sets: self._deque, self._index
        self.contains_finished_episode = False
        self.gamma = gamma
        self.reward_func = reward_function

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
                f"transition_batch must be a TransitionBatch, got: {type(transition_batch)}")

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

        idx = self._sumtree.sample(n=batch_size)
        
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_indices, real_batch_indices = onp.split(idx, [nb_virtual])

        # Create virtual transitions by sampling new desired goals and computing new reward
        virtual_data = self._get_virtual_samples(virtual_batch_indices)
        real_data = _concatenate_leaves(self._storage[real_batch_indices])

        transition_batch = _concatenate_leaves([real_data, virtual_data])
        return transition_batch
    
    
    def _get_virtual_samples(self, batch_indices):
        virtual_batch =[]
        for i in batch_indices:
            done_index = None
            idx = i
            #Find last state in same episode
            # If there is none, just return transition without altering
            while done_index is None and idx < self.capacity:
                if self._storage[idx].done:
                    done_index = idx
                idx += 1

            if done_index is None:
                virtual_batch.append(self._storage[i]) 
            else:
                (state, action, logp, done, gamma, s_next, a_next, logp_next, w, idx, info) = self._storage[i]
                #If goal was reached: do nothing
                if not info[self.goal_keyword]:
                    #else set goal reached in info to true
                    info[self.goal_keyword] = True
                    #adapt reward
                    reward = self.reward_func(state, action, info)
                virtual_batch.append(TransitionBatch.from_single(state, action, logp, reward, done, gamma, s_next, a_next, logp_next,w, idx, info))
        return _concatenate_leaves(virtual_batch)


    def clear(self):
        r""" Clear the experience replay buffer. """
        self._storage = onp.full(shape=(self.capacity,), fill_value=None, dtype='object')
        self._sumtree = SumTree(capacity=self.capacity, random_seed=self._random_seed)
        self._index = 0


    def __len__(self):
        return min(self.capacity, self._index)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._storage[:len(self)])

def _concatenate_leaves(pytrees):
    return jax.tree_map(lambda *leaves: onp.concatenate(leaves, axis=0), *pytrees)