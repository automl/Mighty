class MightyMetaComponent:
    BASE_KEYS = ["env", "vf", "step", "policy"]

    def __init__(self) -> None:
        self.pre_step_methods = []
        self.post_step_methods = []
        self.pre_update_methods = []
        self.post_update_methods = []
        self.pre_episode_methods = []
        self.post_episode_methods = []

    def pre_step(self, metrics):
        for m in self.pre_step_methods:
            m(metrics)

    def post_step(self, metrics):
        for m in self.post_step_methods:
            m(metrics)

    def pre_update(self, metrics):
        for m in self.pre_update_methods:
            m(metrics)

    def post_update(self, metrics):
        for m in self.post_update_methods:
            m(metrics)

    def pre_episode(self, metrics):
        for m in self.pre_episode_methods:
            m(metrics)

    def post_episode(self, metrics):
        for m in self.post_episode_methods:
            m(metrics)
