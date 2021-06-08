import os
from utils.base_config_parser import BaseConfigParser


class DDQNConfigParser(BaseConfigParser):
    def __init__(self, default_config_files: list[str] = []):
        """
        Initialize the scenario configuration parser.

        Parameters
        ----------
        default_config_files: list[str], optional
            When specified, this list of config files will
            be parsed in order, with the values from each config file
            taking precedence over previous ones. This allows an application
            to look for config files in multiple standard locations such as
            the install directory, home directory, and current directory.
            Also, shell * syntax can be used to specify all conf files in a
            directory. For example:
            ["/etc/conf/app_config.ini",
             "/etc/conf/conf-enabled/*.ini",
            "~/.my_app_config.ini",
            "./app_config.txt"]
            The default is an empty list [].
            If it is not specified it looks for the file ROOT/data/configs/default/default_ddqn.ini, where ROOT is
            the package directory.
        """
        if not default_config_files:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # we are in ROOT/utils so we need to go one folder up --> ".."
            default_config_path = os.path.join(dir_path, "..", "data/configs/default")
            default_config_filename = os.path.join(default_config_path, "default_ddqn.ini")
            default_config_files = [default_config_filename]
        super(DDQNConfigParser, self).__init__(default_config_files=default_config_files)
        self._add_arguments()

    def _add_arguments(self):
        """
        Adds DDQN arguments.

        Returns
        -------
        None

        """
        self._p.add_argument(
            '--agent_type',
            default="DDQN",
            type=str,
            choices=["DDQN"],
            help='Name of agent. Read-only.'
        )
        self._p.add_argument(
            '--parser_name',
            default=type(self).__name__,
            choices=[type(self).__name__],
            type=str,
            help='Name of configuration parser. Read-only.'
        )

        # Hyperparameters
        self._p.add_argument(
            '--gamma', '--discount_factor',
            default=0.99,
            type=float,
            help='Discount factor.'
        )
        self._p.add_argument(
            '--epsilon',
            default=0.2,
            type=float,
            help='Controls epsilon-greedy action selection in policy.'
        )
        self._p.add_argument(
            '--max_size_replay_buffer',
            default=1_000_000,
            type=int,
            help="Maximum size of replay buffer."
        )

        # Training
        self._p.add_argument(
            '--learning_rate',
            default=0.001,
            type=int,
            help='Batch size for training.'
        )
        self._p.add_argument(
            '--batch_size',  # TODO: does this belong here or in scenario config parser?
            default=64,
            type=int,
            help='Batch size for training.'
        )
        self._p.add_argument(
            '--begin_updating_weights',
            default=1,
            type=int,
            help="Begin updating policy weights after this many observed transitions."
        )
        self._p.add_argument(
            '--soft_update_weight',
            default=0.01,
            type=float,
            help="???"  # TODO add help
        )
        self._p.add_argument(
            '--max_env_time_steps',
            default=1_000_000,
            type=int,
            help="Maximum number of steps in the environment before episode ends."
        )

