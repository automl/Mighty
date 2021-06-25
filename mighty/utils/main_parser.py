from typing import Optional, Dict
from pathlib import Path
import configargparse
from mighty.utils.scenario_config_parser import ScenarioConfigParser
from mighty.utils.agent_parser import AgentConfigParser
from mighty.iohandling.experiment_tracking import prepare_output_dir


class MainParser(object):
    def __init__(self):
        self.parser_scenario = None  # type: Optional[ScenarioConfigParser]
        self.parser_agent = None
        self.unknown_args = None

        self.scenario_fn = None  # type: Optional[str, Path]
        self.agent_cfg_fn = None  # type: Optional[str, Path]
        self.args_dict = {}  # type: Dict[str, configargparse.Namespace]

    def parse(self, args=None):
        self.parser_scenario = ScenarioConfigParser()  # TODO add master parser :)
        # By using unknown args we can sequentially parse all arguments.
        args, unknown_args = self.parser_scenario.parse()

        self.parser_agent = AgentConfigParser()
        args_agent, unknown_args = self.parser_agent.parse(unknown_args)

        self.unknown_args = unknown_args

        if not args.load_model:
            args.out_dir = prepare_output_dir(
                args,
                user_specified_dir=args.out_dir,
                subfolder_naming_scheme=args.out_dir_suffix
            )

        self.scenario_fn = Path(args.out_dir) / "scenario.ini"  # TODO: where exactly to save this?
        self.agent_cfg_fn = Path(args.out_dir) / "agent.ini"

        print(args_agent)

        args_dict = {
            "scenario": args,
            "agent": args_agent
        }
        self.args_dict = args_dict
        return args_dict
    
    def to_ini(self):
        if self.args_dict:
            self.parser_scenario.to_ini(self.scenario_fn, self.args_dict["scenario"])
            self.parser_agent.to_ini(self.agent_cfg_fn, self.args_dict["agent"])
        else:
            print("Please parse arguments first, nothing to write.")  # TODO log this with logger
