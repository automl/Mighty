from __future__ import annotations

import os
import logging
from omegaconf import OmegaConf
from mighty.utils.logger import Logger, load_logs, to_dataframe


class TestLogging:
    dummy_config = OmegaConf.create({})

    def setup_logger(self):
        logger = Logger(
            experiment_name="test_logger",
            output_path="test_output",
            step_write_frequency=100,
            episode_write_frequency=None,
            hydra_config=self.dummy_config,
            cli_log_lvl=logging.INFO,
        )
        assert not logger.eval, "Logger should not be in eval mode"
        assert logger.total_steps == 0, "Total steps should be 0"
        assert logger.step == 0, "Step should be 0"
        assert logger.episode == 0, "Episode should be 0"
        assert len(logger.buffer) == 0, "Buffer should be empty"
        assert len(logger.current_step.keys()) == 0, "Current step should be empty"
        return logger

    def clean(self, logger):
        logger.close()
        os.remove(logger.log_file.name)
        os.remove(logger.log_dir / "eval.jsonl")
        os.removedirs(logger.log_dir)

    def test_stepping(self):
        logger = self.setup_logger()
        logger.next_step()
        assert logger.step == 1, f"Step should be 1, got {logger.step}"
        logger.next_episode(instance=1)
        assert logger.episode == 1, f"Episode should be 1, got {logger.episode}"
        assert (
            logger.step == 0
        ), f"Step should be 0 after episode end, got {logger.step}"
        logger.reset_episode(instance=1)
        assert (
            logger.episode == 0
        ), f"Episode should be 0 after full logger reset, got {logger.episode}"
        assert (
            logger.step == 0
        ), f"Step should be 0 after full logger reset, got {logger.step}"
        self.clean(logger)

    def test_log(self):
        logger = self.setup_logger()
        logger.log("test", 1)
        assert (
            logger.current_step["test"] == 1
        ), f"Current step should have test=1, got {logger.current_step}"
        logger.log("test", 2)
        assert (
            logger.current_step["test"] == 2
        ), f"Current step should have test=2, got {logger.current_step}"
        logger.next_step()
        assert (
            len(logger.current_step.keys()) == 0
        ), f"Current step should be empty after forwarding, got {logger.current_step}"
        assert (
            len(logger.buffer) == 0
        ), f"Buffer should be empty after forwarding, got {logger.buffer}"
        logger.write()
        assert (
            len(logger.buffer) == 0
        ), f"Buffer should be empty after write, got {logger.buffer}"
        assert os.path.exists(
            logger.log_file.name
        ), "Log file should exist after writing."
        self.clean(logger)

    def test_log_dict(self):
        logger = self.setup_logger()
        logger.log_dict({"test": 1, "test2": 2})
        assert (
            logger.current_step["test"] == 1
        ), f"Current step should have test=1, got {logger.current_step}"
        assert (
            logger.current_step["test2"] == 2
        ), f"Current step should have test2=2, got {logger.current_step}"
        logger.next_episode(instance=1)
        assert (
            len(logger.current_step.keys()) == 0
        ), f"Current step should be empty after forwarding, got {logger.current_step}"
        assert (
            len(logger.buffer) == 1
        ), f"Buffer should have 1 entry after forwarding, got {logger.buffer}"
        logger.write()
        assert (
            len(logger.buffer) == 0
        ), f"Buffer should be empty after write, got {logger.buffer}"
        assert os.path.exists(
            logger.log_file.name
        ), "Log file should exist after writing."
        self.clean(logger)

    def test_get_log_file(self):
        logger = self.setup_logger()
        assert (
            str(logger.get_logfile()) == str(logger.log_file.name)
        ), f"Log file not returned correctly: {logger.get_logfile()} != {logger.log_file.name}"
        self.clean(logger)

    def test_eval(self):
        logger = self.setup_logger()
        logger.set_eval(True)
        assert logger.eval, "Logger should be in eval mode"

    def test_loading(self):
        logger = self.setup_logger()
        logger.log("test", 1)
        logger.write()
        logs = load_logs(logger.log_file.name)
        assert len(logs) == 1, f"Logs should have 1 entry, got {len(logs)}"
        df = to_dataframe(logs)
        assert len(df) == 1, f"Dataframe should have 1 entry, got {len(df)}"
        self.clean(logger)
