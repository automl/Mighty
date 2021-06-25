class RolloutWorker:
    def __init__(self, agent, agent_checkpoint, logger):
        self.agent = agent
        self.agent.load(agent_checkpoint)
        self.logger = logger

    def evaluate(self, env, timesteps):
        print("Starting evaluation")
        for i in range(timesteps):
            done = False
            s = env.reset()
            self.logger.set_eval_env(env)
            while not done:
                a = self.agent.get_action(s, 0)
                ns, r, done, _ = env.step(a)
            self.logger.reset_episode()
            self.logger.write()
