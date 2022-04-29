import json
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict, ChainMap
from datetime import datetime
from functools import reduce
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import Union, Dict, Any, Tuple, List
import logging
import sys

import numpy as np
import pandas as pd

from typing import Callable, Iterable
from mighty.env.env_handling import MIGHTYENV

import wandb
from tensorboard_logger import configure, log_value


def get_standard_logger(
    identifier: str,
    level: int = logging.INFO,
    stream_format: str = "%(asctime)s|%(levelname)s|%(name)s|%(message)s",
):
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


def load_logs(log_file: Path) -> List[Dict]:
    """
    Loads the logs from a jsonl written by any logger.
    The result is the list of dicts in the format:
    {
        'instance': 0,
        'episode': 0,
        'step': 1,
        'example_log_val':  {
            'values': [val1, val2, ... valn],
            'times: [time1, time2, ..., timen],
        }
        ...
    }
    Parameters
    ----------
    log_file: pathlib.Path
        The path to the log file
    Returns
    -------
    [Dict, ...]
    """
    with open(log_file, "r") as log_file:
        logs = list(map(json.loads, log_file))

    return logs


def split(predicate: Callable, iterable: Iterable) -> Tuple[List, List]:
    """
    Splits the iterable into two list depending on the result of predicate.
    Parameters
    ----------
    predicate: Callable
        A function taking an element of the iterable and return Ture or False
    iterable: Iterable
    Returns
    -------
    (positives, negatives)
    """
    positives, negatives = [], []

    for item in iterable:
        (positives if predicate(item) else negatives).append(item)

    return positives, negatives


def flatten_log_entry(log_entry: Dict) -> List[Dict]:
    """
    Transforms a log entry of format like
    {
        'step': 0,
        'episode': 2,
        'some_value': {
            'values' : [34, 45],
            'times':['28-12-20 16:20:53', '28-12-20 16:21:30'],
        }
    }
    into
    [
        { 'step': 0,'episode': 2, 'value': 34, 'time': '28-12-20 16:20:53'},
        { 'step': 0,'episode': 2, 'value': 45, 'time': '28-12-20 16:21:30'}
    ]
    Parameters
    ----------
    log_entry: Dict
        A log entry
    Returns
    -------
    """
    dict_entries, top_level_entries = split(
        lambda item: isinstance(item[1], dict), log_entry.items()
    )
    rows = []
    for value_name, value_dict in dict_entries:
        current_rows = (
            dict(
                top_level_entries
                + [("value", value), ("time", time), ("name", value_name)]
            )
            for value, time in zip(value_dict["values"], value_dict["times"])
        )

        rows.extend(map(dict, current_rows))

    return rows


def list_to_tuple(list_: List) -> Tuple:
    """
    Recursively transforms a list of lists into tuples of tuples
    Parameters
    ----------
    list_:
        (nested) list
    Returns
    -------
    (nested) tuple
    """
    return tuple(
        list_to_tuple(item) if isinstance(item, list) else item for item in list_
    )


def log2dataframe(
    logs: List[dict], wide: bool = False, drop_columns: List[str] = ["time"]
) -> pd.DataFrame:
    """
    Converts a list of log entries to a pandas dataframe.
    Usually used in combination with load_dataframe.
    Parameters
    ----------
    logs: List
        List of log entries
    wide: bool
        wide=False (default) produces a dataframe with columns (episode, step, time, name, value)
        wide=True returns a dataframe (episode, step, time, name_1, name_2, ...) if the variable name_n has not been logged
        at (episode, step, time) name_n is NaN.
    drop_columns: List[str]
        List of column names to be dropped (before reshaping the long dataframe) mostly used in combination
        with wide=True to reduce NaN values
    Returns
    -------
    dataframe
    """

    flat_logs = map(flatten_log_entry, logs)
    rows = reduce(lambda l1, l2: l1 + l2, flat_logs)

    dataframe = pd.DataFrame(rows)
    dataframe.time = pd.to_datetime(dataframe.time)

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
            - set(primary_index_columns + ["time", "value", field_id_column])
        )
        index_columns = primary_index_columns + additional_columns + [field_id_column]
        dataframe = dataframe.set_index(index_columns)
        dataframe = dataframe.unstack()
        dataframe.reset_index(inplace=True)
        dataframe.columns = [a if b == "" else b for a, b in dataframe.columns]

    return dataframe.infer_objects()


class AbstractLogger(metaclass=ABCMeta):
    """
    Logger interface.
    The logger classes provide a way of writing structured logs as jsonl files and also help to track information like
    current episode, step, time ...
    In the jsonl log file each row corresponds to a step.
    """

    valid_types = {
        "recursive": [dict, list, tuple, np.ndarray],
        "primitive": [str, int, float, bool, np.number],
    }

    def __init__(
        self,
        experiment_name: str,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
    ):
        """
        Parameters
        ----------
        experiment_name: str
            Name of the folder to store the result in
        output_path: pathlib.Path
            Path under which the experiment folder is created
        step_write_frequency: int
            number of steps after which the loggers writes to file.
            If None only the data is only written to file if  write is called, if triggered by episode_write_frequency
            or on close
        episode_write_frequency: int
            see step_write_frequency
        """
        self.experiment_name = experiment_name
        self.output_path = os.getcwd()
        self.log_dir = self._init_logging_dir(Path(self.output_path) / self.experiment_name)
        self.step_write_frequency = step_write_frequency
        self.episode_write_frequency = episode_write_frequency
        self.additional_info = {"instance": None}

    @staticmethod
    def _pretty_valid_types() -> str:
        """
        Returns a string pretty string representation of the types that can be logged as values
        Returns
        -------
        """
        valid_types = chain(
            AbstractLogger.valid_types["recursive"],
            AbstractLogger.valid_types["primitive"],
        )
        return ", ".join(map(lambda type_: type_.__name__, valid_types))

    @staticmethod
    def _init_logging_dir(log_dir: Path) -> None:
        """
         Prepares the logging directory
        Parameters
        ----------
        log_dir: pathlib.Path
        Returns
        -------
        None
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def is_of_valid_type(self, value: Any) -> bool:
        f"""
        Checks if the value of any type in {AbstractLogger._pretty_valid_types()}
        Parameters
        ----------
        value
        Returns
        -------
        bool
        """

        if any(isinstance(value, type) for type in self.valid_types["primitive"]):
            return True

        elif any(isinstance(value, type) for type in self.valid_types["recursive"]):
            value = value.vlaues() if isinstance(value, dict) else value
            return all(self.is_of_valid_type(sub_value) for sub_value in value)

        else:
            return False

    @abstractmethod
    def close(self) -> None:
        """
        Makes sure, that all remaining entries in the are written to file and the file is closed.
        Returns
        -------
        """
        pass

    @abstractmethod
    def next_step(self) -> None:
        """
        Call at the end of the step.
        Updates the internal state and dumps the information of the last step into a json
        Returns
        -------
        """
        pass

    @abstractmethod
    def next_episode(self) -> None:
        """
        Call at the end of episode.
        See next_step
        Returns
        -------
        """
        pass

    @abstractmethod
    def write(self) -> None:
        """
        Writes buffered logs to file.
        Invoke manually if you want to load logs during a run.
        Returns
        -------
        """
        pass

    @abstractmethod
    def log(self, key: str, value) -> None:
        f"""
        Writes value to list of values and save the current time for key
        Parameters
        ----------
        key: str
        value:
            the value must of of a type that is json serializable.
            Currently only {AbstractLogger._pretty_valid_types()} and recursive types of those are supported.
        Returns
        -------
        """
        pass

    @abstractmethod
    def log_dict(self, data):
        """
        Alternative to log if more the one value should be logged at once.
        Parameters
        ----------
        data: dict
            a dict with key-value so that each value is a valid value for log
        Returns
        -------
        """
        pass



class Logger(AbstractLogger):
    """
    A logger that manages the creation of the module loggers.
    To get a ModuleLogger for you module (e.g. wrapper) call module_logger = Logger(...).add_module("my_wrapper").
    From now on  module_logger.log(...) or logger.log(..., module="my_wrapper") can be used to log.
    The logger module takes care of updating information like episode and step in the subloggers. To indicate to the loggers
    the end of the episode or the next_step simple call logger.next_episode() or logger.next_step().
    """

    def __init__(
        self,
        experiment_name: str,
        step_write_frequency: int = None,
        episode_write_frequency: int = 1,
        log_to_wandb: str = None,
        log_to_tensorboad: str = None,
        hydra_config=None
    ) -> None:
        """
        Parameters
        ----------
        experiment_name: str
            Name of the folder to store the result in
        output_path: pathlib.Path
            Path under which the experiment folder is created
        step_write_frequency: int
            number of steps after which the loggers writes to file.
            If None only the data is only written to file if  write is called, if triggered by episode_write_frequency
            or on close
        episode_write_frequency: int
            see step_write_frequency
        """
        super(Logger, self).__init__(
            experiment_name, step_write_frequency, episode_write_frequency
        )
        self.log_to_wandb = log_to_wandb
        if log_to_wandb:
            self.run = wandb.init(config=hydra_config, project=self.log_to_wandb)

        self.log_to_tb = log_to_tensorboad
        if log_to_tensorboad:
            configure(Path(self.log_dir) / self.log_to_tb)

        self.instance = None

        self.reward_log_file = open(self.log_dir / f"rewards.jsonl", "w")
        self.eval_log_file = open(self.log_dir / f"eval.jsonl", "w")
        self.log_file = self.reward_log_file
        self.eval = False

        self.step = 0
        self.episode = 0
        self.buffer = []
        self.current_step = {}

    def set_eval(self, eval):
        if eval:
            self.log_file = self.eval_log_file
            self.eval = True
        else:
            self.log_file = self.reward_log_file
            self.eval = False

    def get_logfile(self) -> Path:
        """
        Returns
        -------
        pathlib.Path
            the path to the log file of this logger
        """
        return Path(self.log_file.name)

    @staticmethod
    def __json_default(object):
        """
        Add supoort for dumping numpy arrays and numbers to json
        Parameters
        ----------
        object
        Returns
        -------
        """
        if isinstance(object, np.ndarray):
            return object.tolist()
        elif isinstance(object, np.number):
            return object.item()
        else:
            raise ValueError(f"Type {type(object)} not supported")

    def close(self):
        """
        Makes sure, that all remaining entries (from all sublogger) are written to files and the files are closed.
        Returns
        -------
        """
        for log_file in [self.reward_log_file, self.eval_log_file]:
            if not log_file.closed:
                self.write()
                log_file.close()

    def __del__(self):
        self.close()

    def __end_step(self):
        if self.current_step:
            self.current_step["step"] = self.step
            self.current_step["episode"] = self.episode
            self.current_step["instance"] = self.instance
            self.buffer.append(
                json.dumps(self.current_step, default=self.__json_default)
            )
        self.current_step = {}

    def next_step(self):
        """
        Call at the end of the step.
        Updates the internal state of all subloggers and dumps the information of the last step into a json
        Returns
        -------
        """
        self.__end_step()
        if (
                self.step_write_frequency is not None
                and self.step % self.step_write_frequency == 0
        ):
            self.write()
        if not self.eval:
            self.step += 1

    def next_episode(self, instance):
        """
        Call at the end of episode.
        See next_step
        Returns
        -------
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

        if self.log_to_wandb:
            self.run.log({'instance': instance}, step=self.step)

    def __buffer_to_file(self):
        if len(self.buffer) > 0:
            self.log_file.write("\n".join(self.buffer))
            self.log_file.write("\n")
            self.buffer.clear()
            self.log_file.flush()

    def reset_episode(self, instance):
        self.instance = instance
        self.__end_step()
        self.episode = 0
        self.step = 0

        if self.log_to_wandb is not None:
            self.run.log({'instance': instance}, step=self.step)

    def write(self):
        """
        Writes buffered logs to file.
        Invoke manually if you want to load logs during a run.
        Returns
        -------
        """
        self.__end_step()
        self.__buffer_to_file()

    def __log(self, key, value):
        if not self.is_of_valid_type(value):
            valid_types = self._pretty_valid_types()
            raise ValueError(
                f"value {type(value)} is not of valid type or a recursive composition of valid types ({valid_types})"
            )
        self.current_step[key] = value

    def log(
        self, key: str, value: Union[Dict, List, Tuple, str, int, float, bool]
    ) -> None:
        f"""
        Writes value to list of values and save the current time for key
        Parameters
        ----------
        key: str
        value:
           the value must of of a type that is json serializable.
           Currently only {AbstractLogger._pretty_valid_types()} and recursive types of those are supported.
        Returns
        -------
        """
        self.__log(key, value)
        if self.log_to_wandb:
            self.run.log({key: value}, step=self.step)

        if self.log_to_tb:
            # This only logs floats, so we skip the rest
            try:
                log_value(key, value, self.step)
            except:
                pass

    def log_dict(self, data: Dict) -> None:
        """
        Alternative to log if more than one value should be logged at once.
        Parameters
        ----------
        data: dict
            a dict with key-value so that each value is a valid value for log
        Returns
        -------
        """
        for key, value in data.items():
            self.__log(key, value)

        if self.log_to_wandb:
            self.run.log(data, step=self.step)

        if self.log_to_tb:
            for key, value in data.items():
                log_value(key, value, self.step)