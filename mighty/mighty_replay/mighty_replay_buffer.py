from coax.experience_replay._simple import SimpleReplayBuffer
from coax.experience_replay._prioritized import PrioritizedReplayBuffer

class MightyReplay(SimpleReplayBuffer):
    def add(self, transition_batch, metrics):
        super().add(transition_batch)


class PrioritizedReplay(PrioritizedReplayBuffer):
    def add(self, transition_batch, metrics):
        advantage = metrics['td_error']
        super().add(transition_batch, advantage)