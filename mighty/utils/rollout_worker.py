class RolloutWorker:
    def __init__(self, agent, agent_checkpoint, logger):
        self.agent = agent
        self.agent.load(agent_checkpoint)
        self.logger = logger

    def evaluate(self, env, timesteps):
        #import gym
        #env = gym.make("Pendulum-v0")
        print("Starting evaluation")
        reward = []
        for i in range(timesteps):
            done = False
            s = env.reset()
            self.logger.reset_episode()
            self.logger.set_env(env)
            rew = 0
            while not done:
                a = self.agent.get_action(s, epsilon=0)
                ns, r, done, _ = env.step(a)
                rew += r
            reward.append(rew)
            self.logger.write()
        print(f"Eval reward:{sum(reward)/len(reward)}")

