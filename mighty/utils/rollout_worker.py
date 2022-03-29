class RolloutWorker:
    def __init__(self, agent, agent_checkpoint, logger):
        self.agent = agent
        self.agent.load_checkpoint(agent_checkpoint)
        self.logger = logger

    def evaluate(self, env, episodes):
        print("Starting evaluation")
        for i in range(episodes):
            done = False
            s = env.reset()
            self.logger.set_eval_env(env)
            while not done:
                a = self.agent.get_action(s, epsilon=0)
                ns, r, done, _ = env.step(a)
            self.logger.reset_episode()
            self.logger.write()
