from typing import Optional, Dict, List
from pathlib import Path
import configargparse
from mighty.utils.scenario_config_parser import ScenarioConfigParser
from mighty.utils.agent_parser import AgentConfigParser
from mighty.iohandling.experiment_tracking import prepare_output_dir


class MainParser(object):
    def __init__(self):
        """
        Initialize parser.
        The parser for the scenario and the agent are created via the parse call.
        """
        self.parser_scenario = None  # type: Optional[ScenarioConfigParser]
        self.parser_agent = None  # type: Optional[AgentConfigParser]
        self.unknown_args = None  # type: Optional[List[str]]

        self.scenario_fn = None  # type: Optional[str, Path]
        self.agent_cfg_fn = None  # type: Optional[str, Path]
        self.args_dict = {}  # type: Dict[str, configargparse.Namespace]

    def parse(self, args: Optional[List[str]] = None):
        """
        Parse command line arguments and prepare output directory.

        Instantiate parsers and pare arguments in following order:
        1. parse scenario arguments
        2. parse agent arguments

        :param args: optional command line arguments.

        :return: Dictionary containing the arguments for the scenario and the agent as namespaces. Keys: ["scenario", "agent"].
        """

        # 1. Parse scenario arguments.
        self.parser_scenario = ScenarioConfigParser()
        args, unknown_args = self.parser_scenario.parse(args=args)

        # 2. Parse agent arguments.
        self.parser_agent = AgentConfigParser()
        args_agent, unknown_args = self.parser_agent.parse(unknown_args)

        # Save for later handling.
        self.unknown_args = unknown_args

        # Prepare output directory and config files
        if not args.load_model:
            args.out_dir = prepare_output_dir(
                args,
                user_specified_dir=args.out_dir,
                subfolder_naming_scheme=args.out_dir_suffix
            )
        self.scenario_fn = Path(args.out_dir) / "scenario.ini"  # TODO: where exactly to save this?
        self.agent_cfg_fn = Path(args.out_dir) / "agent.ini"  # TODO: if we load a model, do we want to overwrite the configs? Can the configs differ?

        args_dict = {
            "scenario": args,
            "agent": args_agent
        }
        self.args_dict = args_dict
        return args_dict

    def to_ini(self):
        """
        Write the scenario and the agent arguments to an ini-file.
        :return:
        """
        if self.args_dict:
            self.parser_scenario.to_ini(self.scenario_fn, self.args_dict["scenario"])
            self.parser_agent.to_ini(self.agent_cfg_fn, self.args_dict["agent"])
        else:
            print("Please parse arguments first, nothing to write.")  # TODO log this with logger or throw exception?
