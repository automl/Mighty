class MightyMetaComponent:
    METRICS = {'dqn': [], 'ppo': [], 'sac': []}
    def __init__(self, algo) -> None:
        self.metrics = self.METRICS[algo]
        self.pre_step_methods = []
        self.post_step_methods = []
        self.pre_update_methods = []
        self.post_update_methods = []
        self.pre_episode_methods = []
        self.post_episode_methods = []

    def pre_step(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.pre_step_methods:
            m(**args)

    def post_step(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.post_step_methods:
            m(**args)

    def pre_update(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.pre_update_methods:
            m(**args)

    def post_update(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.post_update_methods:
            m(**args)

    def pre_episode(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.pre_episode_methods:
            m(**args)

    def post_episode(self, metrics):
        args = [metrics[k] for k in self.metrics]
        for m in self.post_episode_methods:
            m(**args)