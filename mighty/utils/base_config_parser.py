from pathlib import Path
import configargparse


class BaseConfigParser(object):
    def __init__(self, default_config_files: list[str] = []):
        """
        Initialize the configuration parser.

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
        """
        self._default_cfg_dir = "."
        self._default_cfg_fns = default_config_files
        self._p = configargparse.ArgParser(
            default_config_files=self._default_cfg_fns,
            config_file_parser_class=configargparse.ConfigparserConfigFileParser
        )
        self.opts = None
        self.unknown_args = None

    def parse(self, args: list[str] = None, config_filename: str = ""):
        """
        Parse given arguments and/or a config file.

        If neither args nor config_filename are given, the default arguments defined in the parser are used.
        If a default_config_files are specified, the default arguments from the parser are overwritten.
        If the config_filename is given, the arguments in there are used.
        If args are given, args will be used.
        If both config_filename and args are given, the configuration in config_filename will be parsed first, then
        args.
        So the final priority is:
            default arguments from parser < arguments from default configuration file < arguments from custom config
            file < command line arguments.
        In short:
            argparse < default cfg file < custom cfg file < cmd line args.

        Parameters
        ----------
        args: list[str], optional
            Command line arguments. The default is None.
        config_filename: str, optional
            Filename of configuration ini-file.

        Returns
        -------
        argparse.Namespace
            Parsed arguments.

        """
        config_file_contents = None
        if config_filename:
            with open(config_filename, 'r') as file:
                config_file_contents = file.read()
        self.opts, unknown_args = self._p.parse_known_args(args=args, config_file_contents=config_file_contents)
        self.unknown_args = unknown_args
        return self.opts, self.unknown_args

    def to_ini(self, filename: str, namespace: configargparse.Namespace = None):
        if not namespace and not self.opts:
            msg = "Please parse args or pass a namespace to save configuration to disk."
            raise ValueError(msg)

        opts_to_disk = namespace if namespace else self.opts
        output_file_paths = [filename]
        for fp in output_file_paths:
            fp = Path(fp)
            fp.parent.mkdir(parents=True, exist_ok=True)
        output_file_paths = [str(p) for p in output_file_paths]  # configargparse does not like PosixPaths for printing
        self._p.write_config_file(parsed_namespace=opts_to_disk, output_file_paths=output_file_paths)
