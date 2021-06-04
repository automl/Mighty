from mighty.env.env_handling import DACENV

class MockEnv(DACENV):

    def step(self, action):
        pass

    def reset(self):
        pass
