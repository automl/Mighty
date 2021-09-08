import numpy as np

class RolloutWorker:
    def __init__(self, agent, agent_checkpoint, logger):
        self.agent = agent
        self.agent.load(agent_checkpoint)
        self.logger = logger

    def evaluate(self, env, timesteps):
        print("Starting evaluation")
        avg_reward = 0.
        for _ in range(timesteps):
            state, done = env.reset(), False
            self.logger.reset_episode()
            self.logger.set_env(env)
            while not done:
                action = self.agent.get_action(np.array(state), epsilon=0)
                state, reward, done, _ = env.step(action)
                avg_reward += reward
            self.logger.write()
        avg_reward /= episodes
        print(f"Eval reward:{avg_reward}")

