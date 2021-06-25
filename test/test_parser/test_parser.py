import unittest
import os
import numpy as np

from mighty.utils.scenario_config_parser import ScenarioConfigParser
from mighty.utils.base_config_parser import BaseConfigParser


class ParserTestCase(unittest.TestCase):
    def test_scenario_parser_from_customfile(self):
        scenario_parser = ScenarioConfigParser()
        fn = "mighty/data/configs/custom/custom_train.ini"
        args = None
        args, unknown_args = scenario_parser.parse(args=args, config_filename=fn)
        self.assertTrue(args.episodes, 30)

    def test_default_trainfile_existance(self):
        fn = "mighty/data/configs/custom/custom_train.ini"
        self.assertTrue(os.path.isfile(fn))

    def test_scenario_parser_from_defaultfile(self):
        scenario_parser = ScenarioConfigParser()
        fn = ""
        args = None
        args, unknown_args = scenario_parser.parse(args=args, config_filename=fn)
        self.assertTrue(args.episodes, 100)

    def test_scenario_parser_from_customfile_plus_cmdline(self):
        scenario_parser = ScenarioConfigParser()
        fn = "mighty/data/configs/custom/custom_train.ini"
        args = ["--episodes", "42"]  # simulate command line args
        args, unknown_args = scenario_parser.parse(args=args, config_filename=fn)
        self.assertTrue(args.episodes, 42)

    def test_scenario_parser_unknownargs(self):
        scenario_parser = ScenarioConfigParser()
        fn = ""
        args_in = ["--agent_type", "DDQN"]  # simulate command line args
        args, unknown_args = scenario_parser.parse(args=args_in, config_filename=fn)
        print(unknown_args, args_in)
        self.assertTrue(unknown_args == args_in)

    def test_scenario_parser_write_ini(self):
        scenario_parser = ScenarioConfigParser()
        fn = ""
        args = ["--episodes", "42"]  # simulate command line args
        args, unknown_args = scenario_parser.parse(args=args, config_filename=fn)
        fn_out = "mighty/test/test_parser/custom_train_new.ini"
        if os.path.isfile(fn_out):
            os.remove(fn_out)
        scenario_parser.to_ini(filename=fn_out, namespace=args)
        self.assertTrue(os.path.isfile(fn_out))

    def test_baseparser_to_ini(self):
        parser = BaseConfigParser()
        self.assertRaises(ValueError, parser.to_ini, "sdfsdfsd")
