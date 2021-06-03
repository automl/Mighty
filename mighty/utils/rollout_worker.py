class RolloutWorker:
    def __init__(self, agent, agent_checkpoint):
        self.agent = agent
        agent.load(agent_checkpoint)

    def evaluate(self, env, timesteps, logger):
        for i in range(timesteps):
            done = False
            s = env.reset()
            self.logger.reset_episode()
            self.logger.set_env(self.env)
            while not done:
                a = self.agent.get_action(s)
                ns, r, done, _ = env.step(a)
            logger.next_episode()
