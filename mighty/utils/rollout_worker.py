import numpy as np

class RolloutWorker:
    def __init__(self, agent, agent_checkpoint, logger):
        self.agent = agent
        self.agent.load_checkpoint(agent_checkpoint)
        self.logger = logger

    def evaluate(self, env, episodes):
        print("Starting evaluation")
        avg_reward = 0.
        for _ in range(episodes):
            state, done = env.reset(), False
            print(f"Reset: {state}")
            self.logger.reset_episode()
            while not done:
                action = self.agent.get_action(np.array(state), epsilon=0)
                state, reward, done, _ = env.step(action)
                print(action)
                print(state)
                avg_reward += reward
            self.logger.write()
        avg_reward /= episodes
        print(f"Eval reward:{avg_reward}")

