import logging

import numpy as np
from datetime import datetime
from pathlib import Path
import time
import sys
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger

from mighty.utils.logger import get_standard_logger


class DummyWorker(object):
    def __init__(self):
        # self.identifier = np.random.randint(0, 100000)
        self.var = 3

    def work(self, multiplicand: int):
        res = self.var * multiplicand
        # print(self.identifier)
        return res


class DummyEvaluator(object):
    def __init__(self,):
        pass

    def evaluate(self):
        instances = np.arange(0, 100)


def training(local_rank, config):
    t_start = time.time()
    rank = idist.get_rank()

    # logger = setup_logger(name=f"Dummy Evaluator rank {rank}")
    logger = get_standard_logger(identifier=f"Dummy Evaluator rank {rank}")
    logger.setLevel(logging.INFO)
    # log_basic_info(logger, config)
    # if rank == 0:
    #     device = idist.device()
    #     backend = idist.backend()
    #     world_size = idist.get_world_size()
    #     logger.info(f"backend {backend}, world size {world_size}")
    #
    #     if "cuda" in device.type:
    #         config["cuda device name"] = torch.cuda.get_device_name(local_rank)
    #         logger.info(f"cuda device name {config['cuda device name']}")
    # logger.info(f"rank {rank}, local rank {local_rank}")

    # worker = config["worker"]

    # ret = worker.work(multiplicand=local_rank)
    worker_fn = config["worker_fn"]
    ret = worker_fn(multiplicand=local_rank)
    logger.info(f"worker ret {ret}")

    result = idist.all_gather(ret)
    if local_rank == 0:
        fname = "testresults.json"
        # idist.barrier()
        with open(fname, 'w') as file:
            json.dump(result, file)

    t_end = time.time()
    print(f"{local_rank}: {t_end - t_start:.4f}s")


def run(
    backend=None,
    nproc_per_node=None,
    with_amp=False,
    **spawn_kwargs,
):
    """Main entry to train an model on CIFAR10 dataset.
    Args:
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        with_amp (bool): if True, enables native automatic mixed precision. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes
    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    device = idist.device()

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    # idist.spawn(backend, training, args=(), kwargs_dict={"config": config}, nproc_per_node=nproc_per_node)

    worker = DummyWorker()
    config["worker_fn"] = worker.work
    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)
    print("done")
    idist.finalize()  # end all subprocesses


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


if __name__ == '__main__':
    run(
        backend="gloo",
        nproc_per_node=4
    )
