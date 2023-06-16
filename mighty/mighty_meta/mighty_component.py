class MightyMetaComponent:
    """Component for registering meta-control methods."""

    def __init__(self) -> None:
        """
        Meta module init.

        :return:
        """

        self.pre_step_methods = []
        self.post_step_methods = []
        self.pre_update_methods = []
        self.post_update_methods = []
        self.pre_episode_methods = []
        self.post_episode_methods = []

    def pre_step(self, metrics):
        """
        Execute methods before a step.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.pre_step_methods:
            m(metrics)

    def post_step(self, metrics):
        """
        Execute methods after a step.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.post_step_methods:
            m(metrics)

    def pre_update(self, metrics):
        """
        Execute methods before the update.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.pre_update_methods:
            m(metrics)

    def post_update(self, metrics):
        """
        Execute methods after the update.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.post_update_methods:
            m(metrics)

    def pre_episode(self, metrics):
        """
        Execute methods before an episode.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.pre_episode_methods:
            m(metrics)

    def post_episode(self, metrics):
        """
        Execute methods at the end of an episode.

        :param metrics: Current metrics dict
        :return:
        """

        for m in self.post_episode_methods:
            m(metrics)
