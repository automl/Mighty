"""Logger module."""

from __future__ import annotations

import json
import logging
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich.logging import RichHandler

# FIXME: we need to test if this thing actually produces readable logs
# If not, we might want to simplify it a lot and/or rely on tb's logging


def get_standard_logger(
    identifier: str,
    level: int = logging.INFO,
    stream_format: str = "%(asctime)s|%(levelname)s|%(name)s|%(message)s",
):
    """Get logger.

    :param identifier: logger name
    :param level: logger level
    :param stream_format: save file naming format
    :return: logger
    """
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(stream_format)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level=level)
    stream_handler.setFormatter(formatter)

    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger


def load_logs(log_file: Path) -> list[dict]:
    """Load logs.

    :param log_file: file to load
    :return: logs
    """
    with open(log_file) as log_file:
        return list(map(json.loads, log_file))


def split(predicate: Callable, iterable: Iterable) -> tuple[list, list]:
    """Splits the iterable into two list depending on the result of predicate.

    :param predicate: A function taking an element of the iterable
        and return Ture or False
    :param iterable: Iterable
    :return: (positives, negatives)
    """
    positives, negatives = [], []

    for item in iterable:
        (positives if predicate(item) else negatives).append(item)

    return positives, negatives


def flatten_log_entry(log_entry: dict) -> list[dict]:
    """Transforms a log entry of format.

    {'step': 0, 'episode': 2,
    'some_value': {'values' : [34, 45], '
        times':['28-12-20 16:20:53', '28-12-20 16:21:30'],}}
    into
    [{ 'step': 0,'episode': 2, 'value': 34, 'time': '28-12-20 16:20:53'},
    { 'step': 0,'episode': 2, 'value': 45, 'time': '28-12-20 16:21:30'}]

    :param log_entry: A log entry as Dict
    :return:
    """
    dict_entries, top_level_entries = split(
        lambda item: isinstance(item[1], dict), log_entry.items()
    )
    rows = []
    for value_name, value_dict in dict_entries:
        current_rows = (
            dict(
                [
                    *top_level_entries,
                    ("value", value),
                    ("time", time),
                    ("name", value_name),
                ]
            )
            for value, time in zip(
                value_dict["values"], value_dict["times"], strict=False
            )
        )

        rows.extend(map(dict, current_rows))

    return rows


def list_to_tuple(list_: list) -> tuple:
    """Recursively transforms a list of lists into tuples of tuples.

    :param list_: (nested) list
    :return: tuple
    """
    return tuple(
        list_to_tuple(item) if isinstance(item, list) else item for item in list_
    )


def to_dataframe(
    logs: list[dict],
    wide: bool = False,
    single_indicators: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Converts a list of log entries to a pandas dataframe.

    Usually used in combination with load_dataframe.

    :param logs: List of log entries
    :param wide: wide=False (default) produces a dataframe with columns
        (episode, step, time, name, value);
        wide=True returns a dataframe
        (episode, step, time, name_1, name_2, ...)
        if the variable name_n has not been logged at
        (episode, step, time) name_n is NaN.
    :param drop_columns: List of column names to be dropped
        (before reshaping the long dataframe) mostly used in combination
        with wide=True to reduce NaN values
    :return: dataframe
    """
    if single_indicators is None:
        single_indicators = ["action", "terminated", "reward"]

    def separate_entries(log):
        num_entries = None
        for k in log:
            if k in single_indicators and isinstance(log[k], list):
                num_entries = len(log[k])
                break

        if num_entries is not None:
            new_logs = []
            for n in range(num_entries):
                new_log = {}
                for k in log:
                    if isinstance(log[k], list):
                        if len(log[k]) == num_entries:
                            new_log[k] = log[k][n]
                        else:
                            new_log[k] = log[k]
                    else:
                        new_log[k] = log[k]
                new_logs.append(new_log)
            log = new_logs
        return log

    separate_logs = list(map(separate_entries, logs))
    all_logs = []
    for log in separate_logs:
        if isinstance(log, list):
            all_logs.extend(log)
        else:
            all_logs.append(log)
    dataframe = pd.DataFrame(all_logs)

    if drop_columns is not None:
        dataframe = dataframe.drop(columns=drop_columns)

    dataframe = dataframe.infer_objects()
    list_column_candidates = dataframe.dtypes == object

    for i, candidate in enumerate(list_column_candidates):
        if candidate:
            dataframe.iloc[:, i] = dataframe.iloc[:, i].apply(
                lambda x: list_to_tuple(x) if isinstance(x, list) else x
            )

    if wide:
        primary_index_columns = ["episode", "step"]
        field_id_column = "name"
        additional_columns = list(
            set(dataframe.columns)
            - {*primary_index_columns, "time", "value", field_id_column}
        )
        index_columns = primary_index_columns + additional_columns + [field_id_column]
        dataframe = dataframe.set_index(index_columns)
        dataframe = dataframe.pivot_table()
        dataframe = dataframe.reset_index()
        dataframe.columns = [a if b == "" else b for a, b in dataframe.columns]

    return dataframe.infer_objects()


class AbstractLogger(metaclass=ABCMeta):
    """Logger interface.

    The logger classes provide a way of writing structured logs as jsonl
    and also help to track information like current episode, step, time ...
    In the jsonl log file each row corresponds to a step.
    """

    valid_types: ClassVar = {
        "recursive": [dict, list, tuple, np.ndarray],
        "primitive": [str, int, float, bool, np.number],
    }

    def __init__(
        self,
        experiment_name: str,
        output_path: str,
        step_write_frequency: int | None = None,
        episode_write_frequency: int = 1,
    ):
        """Initialize.

        :param experiment_name: Name of the folder to store the result in
        :param output_path: Path under which the experiment folder
            is created is pathlib.Path
        :param step_write_frequency: number of steps after
            which the loggers writes to file.
            If None only the data is only written to file if  write is called,
            if triggered by episode_write_frequency or on close
        :param episode_write_frequency: see step_write_frequency
        """
        self.experiment_name = experiment_name
        self.output_path = output_path
        self.log_dir = self._init_logging_dir(
            Path(self.output_path) / self.experiment_name
        )
        self.step_write_frequency = step_write_frequency
        self.episode_write_frequency = episode_write_frequency
        self.additional_info = {"instance": None}

    @staticmethod
    def _pretty_valid_types() -> str:
        """Returns a string pretty string representation
        of the types that can be logged as values.
        """
        valid_types = chain(
            AbstractLogger.valid_types["recursive"],
            AbstractLogger.valid_types["primitive"],
        )
        return ", ".join(type_.__name__ for type_ in valid_types)

    @staticmethod
    def _init_logging_dir(log_dir: Path) -> Path:
        """Prepares the logging directory.

        :param log_dir: Parameters
        :return:
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def is_of_valid_type(self, value: Any) -> bool:
        """Checks if the value of any type in AbstractLogger._pretty_valid_types().

        :param value:
        :return: bool
        """
        is_valid = False
        if (
            any(isinstance(value, vtype) for vtype in self.valid_types["primitive"])
            or value is None
        ):
            is_valid = True
        elif any(isinstance(value, vtype) for vtype in self.valid_types["recursive"]):
            value = value.values() if isinstance(value, dict) else value
            is_valid = all(self.is_of_valid_type(sub_value) for sub_value in value)
        return is_valid

    @abstractmethod
    def close(self) -> None:
        """Makes sure, that all remaining entries in the are
        written to file and the file is closed.

        :return:
        """

    @abstractmethod
    def next_step(self) -> None:
        """Call at the end of the step.

        Updates the internal state and dumps the information
        of the last step into a json

        :return:
        """

    @abstractmethod
    def next_episode(self) -> None:
        """Call at the end of episode. See next_step.

        :return:
        """

    @abstractmethod
    def write(self) -> None:
        """Writes buffered logs to file.

        Invoke manually if you want to load logs during a run.

        :return:
        """

    @abstractmethod
    def log(self, key: str, value) -> None:
        """Writes value to list of values and save the current time for key.

        :param key: keyword to log
        :param value: the value must of of a type that is json serializable.
            Currently only AbstractLogger._pretty_valid_types()
            and recursive types of those are supported.
        :return:
        """

    @abstractmethod
    def log_dict(self, data):
        """Alternative to log if more the one value should be logged at once.

        :param data: a dict with key-value so that each value is a valid value for log
        :return:
        """


class Logger(AbstractLogger, logging.Logger):
    """A logger that manages the creation of the module loggers.

    To get a ModuleLogger for you module (e.g. wrapper)
    call module_logger = Logger(...).add_module("my_wrapper").
    From now on  module_logger.log(...) or
    logger.log(..., module="my_wrapper") can be used to log.
    The logger module takes care of updating information
    like episode and step in the subloggers.
    To indicate to the loggers the end of the episode or the next_step simply
    call logger.next_episode() or logger.next_step().
    """

    def __init__(
        self,
        experiment_name: str,
        output_path: str,
        step_write_frequency: int | None = None,
        episode_write_frequency: int = 1,
        hydra_config=None,
        cli_log_lvl=logging.NOTSET,
    ) -> None:
        """Init Logger.

        :param experiment_name: Name of the folder to store the result in
        :param output_path: Path under which the experiment folder
            is created as pathlib.Path
        :param step_write_frequency: number of steps after
            which the loggers writes to file.
            If None only the data is only written to file if  write is called,
            if triggered by episode_write_frequency or on close
        :param episode_write_frequency: see step_write_frequency
        :return:
        """
        logging.Logger.__init__(self, name="MightyLogger")
        self.addHandler(RichHandler(level=cli_log_lvl))
        AbstractLogger.__init__(
            self,
            experiment_name,
            output_path,
            step_write_frequency,
            episode_write_frequency,
        )

        self.instance = None

        self.reward_log_file = open(self.log_dir / "rewards.jsonl", "w")
        self.eval_log_file = open(self.log_dir / "eval.jsonl", "w")
        self.log_file = self.reward_log_file
        self.eval = False

        self.total_steps = 0
        self.step = 0
        self.episode = 0
        self.buffer = []
        self.current_step = {}
        if OmegaConf.is_config(hydra_config):
            hydra_config = OmegaConf.to_container(hydra_config)
        if hydra_config is not None:
            self.log_dict(hydra_config)

    def set_eval(self, evaluate):
        """Switch to eval mode."""
        if evaluate:
            self.log_file = self.eval_log_file
            self.eval = True
        else:
            self.log_file = self.reward_log_file
            self.eval = False

    def get_logfile(self) -> Path:
        """Return logfile.

        :param Path: the path to the log file of this logger (as pathlib.Path)
        :return:
        """
        return Path(self.log_file.name)

    @staticmethod
    def __json_default(ob):
        """Add supoort for dumping numpy arrays and numbers to json.

        :param object:
        :return:
        """
        if isinstance(ob, np.ndarray):
            jsonable = ob.tolist()
        elif isinstance(ob, np.number):
            jsonable = ob.item()
        else:
            raise ValueError(f"Type {type(ob)} not supported")
        return jsonable

    def close(self):
        """Makes sure, that all remaining entries (from all sublogger)
        are written to files and the files are closed.

        :return:
        """
        for log_file in [self.reward_log_file, self.eval_log_file]:
            if not log_file.closed:
                self.write()
                log_file.close()

    def __del__(self):
        """Delete."""
        self.close()

    def __end_step(self):
        """Cleanup after step."""
        if self.current_step:
            self.current_step["step"] = self.step
            self.current_step["episode"] = self.episode
            self.current_step["total_steps"] = self.total_steps
            self.current_step["instance"] = self.instance
            self.buffer.append(
                json.dumps(self.current_step, default=self.__json_default)
            )
        self.current_step = {}

    def next_step(self):
        """Call at the end of the step.

        Updates the internal state of all subloggers and
        dumps the information of the last step into a json.

        :return:
        """
        self.__end_step()
        if (
            self.step_write_frequency is not None
            and self.total_steps % self.step_write_frequency == 0
        ):
            self.write()

        if not self.eval:
            self.step += 1
            self.total_steps += 1

    def next_episode(self, instance):
        """Call at the end of episode. See next_step.

        :return:
        """
        self.__end_step()
        self.step = 0
        self.instance = instance
        if (
            self.episode_write_frequency is not None
            and self.episode % self.episode_write_frequency == 0
        ):
            self.write()
        if not self.eval:
            self.episode += 1

    def __buffer_to_file(self):
        """Write buffer to file.

        :return:
        """
        if len(self.buffer) > 0:
            self.log_file.write("\n".join(self.buffer))
            self.log_file.write("\n")
            self.buffer.clear()
            self.log_file.flush()

    def reset_episode(self, instance):
        """Reset.

        :param instance: next instance
        :return:
        """
        self.instance = instance
        self.__end_step()
        self.episode = 0
        self.step = 0

    def write(self):
        """Writes buffered logs to file.
        Invoke manually if you want to load logs during a run.

        :return:
        """
        self.__end_step()
        self.__buffer_to_file()

    def __log(self, key, value):
        if not self.is_of_valid_type(value):
            valid_types = self._pretty_valid_types()
            raise ValueError(
                f"""value {type(value)} is not of valid type
                or a recursive composition of valid types ({valid_types})"""
            )
        self.current_step[key] = value

    def log(
        self, key: str, value: dict | list | tuple | str | int | float | bool
    ) -> None:
        """Writes value to list of values and save the current time for key.

        :param key: keyword to log
        :param value: the value must of of a type that is json serializable.
            Currently only AbstractLogger._pretty_valid_types()
            and recursive types of those are supported.
        :return:
        """
        self.__log(key, value)

    def log_dict(self, data: dict) -> None:
        """Alternative to log if more than one value should be logged at once.

        :param data: a dict with key-value so that each value is a valid value for log
        :return:
        """
        for key, value in data.items():
            self.__log(key, value)
