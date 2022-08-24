import os
from .base_config_parser import BaseConfigParser


class ScenarioConfigParser(BaseConfigParser):
    def __init__(self, default_config_files: list[str] = []):
        """
        Initialize the scenario configuration parser.

        :param default_config_files: When specified, this list of config files will
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
            If it is not specified it looks for the file ROOT/data/configs/default/default_train.ini, where ROOT is
            the package directory.
        """
        if not default_config_files:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # we are in ROOT/utils so we need to go one folder up --> ".."
            default_config_path = os.path.join(dir_path, "..", "data/configs/default")
            default_config_filename = os.path.join(default_config_path, "default_train.ini")
            default_config_files = [default_config_filename]
        super(ScenarioConfigParser, self).__init__(default_config_files=default_config_files)
        self._add_arguments()

    def _add_arguments(self):
        """
        Adds scenario/train specific and top-level arguments.
        :return:
        """
        # Training
        self._p.add_argument(
            '--episodes', '-e',
            default=100,
            type=int,
            help='Number of training episodes.'
        )
        self._p.add_argument(
            '--seed', '-s',
            default=12345,
            type=int,
            help='Seed'
        )
        self._p.add_argument(
            '--max-train-steps', '-t',  # TODO should this be called training_steps?
            default=1_000_000,
            type=int,
            help='(Maximum) number of training steps.'
        )

        # Output/Saving
        # TODO discuss output dir structure / how to save what
        self._p.add_argument(
            '--out-dir',
            default="tmp",
            type=str,
            help='Directory to save results. Defaults to tmp dir.'
        )
        self._p.add_argument(
            '--out-dir-suffix',
            default='seed',
            type=str,
            choices=['seed', 'time'],
            help='Created suffix of directory to save results.'
        )
        self._p.add_argument(
            '--eval_every_n_steps',
            default=1_000,
            type=int,
            help='After how many steps to evaluate'
        )
        self._p.add_argument(
            '--load-model',
            default=None
        )  # TODO: add help and type
        self._p.add_argument(
            '--save_model_every_n_episodes',
            default=100,
            type=int,
            help="Save modevery n episodes."
        )

