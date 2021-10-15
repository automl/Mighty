import os
from typing import List, Optional, Dict
import json
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn

import ignite
import ignite.distributed as idist

from mighty.train.parallel.rollout import EvaluationRolloutWorker


class Evaluator(object):
    def __init__(
            self,
            checkpoint_dir: str,
            device: str,
            output_file_name: str,
            env,
            instances_to_evaluate: List[str],
            n_workers: int = 1,
            n_episodes_per_instance: int = 1,
            backend=None,
            nproc_per_node=None,
            with_amp=False,
            watch_mode: bool = False,  # watches if new checkpoints are arriving, should we do this or later?
            spawn_kwargs: Dict = {},
    ):
        """

        :param checkpoint_dir:
        :param device:
        :param output_file_name:
        :param env:
        :param instances_to_evaluate:
        :param n_workers:
        :param n_episodes_per_instance:
        :param backend:
        :param nproc_per_node:
        :param with_amp:
        :param watch_mode:
        :param spawn_kwargs:

        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
        "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
        when main python process is spawning training as child processes.
        with_amp (bool): if True, enables native automatic mixed precision. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes
        """
        self.checkpoint_dir = checkpoint_dir
        if n_workers < 1 or type(n_workers) is not int:
            raise ValueError("n_workers must be >= 1 and of type int.")
        self.n_workers = n_workers
        if n_episodes_per_instance < 1 or type(n_episodes_per_instance) is not int:
            raise ValueError("n_episodes_per_instance must be >= 1 and of type int.")
        devices = ["cuda", "cpu"]
        if device not in devices:
            raise ValueError(f"device must be in {devices}, not {device}.")
        self.device = device
        # TODO figure out appropriate backend for device

        self.n_episodes_per_instance = n_episodes_per_instance
        self.instances = instances_to_evaluate  # only evaluate on those instances from the training set
        self.visited_checkpoint_filenames = []  # add filenames of evaluated checkpoints here

        # TODO: check the existing output file to see if we already validated things
        self.output_file = output_file_name
        if os.path.isfile(output_file_name):  # we can ignore already evaluated checkpoints
            with open(output_file_name, 'r') as out_fh:
                for line in out_fh:
                    data = json.loads(line)
                self.visited_checkpoint_filenames.append(data['checkpoint_path'])

        # TODO: read in checkpoint filenames
        self.checkpoint_data = []
        validation_file = os.path.join(self.checkpoint_dir, 'validation.json')
        with open(validation_file, 'r') as in_fh:
            for line in in_fh:
                data = json.loads(line)
            if data['checkpoint_path'] not in self.visited_checkpoint_filenames:
                self.checkpoint_data.append(data)

        self.total_eval_runs_required = len(self.checkpoint_data) * len(self.instances)
        self.env = env


        # TODO: get agent type
        # TODO: get architecture
        self.architecture = None  # type: nn.Module
        self.policy_type = "continuous"  # TODO make function: determine policy type (discrete or continuous)
        # TODO: get all train instances
        # TODO: setup logging / logfile
        # TODO: setup idist
        self.backend = backend
        self.nproc_per_node = nproc_per_node
        self.with_amp = with_amp
        self.watch_mode = watch_mode
        self.spawn_kwargs = spawn_kwargs

    def evaluate(self):
        config = locals()
        config.update(config["spawn_kwargs"])
        del config["spawn_kwargs"]

        device = idist.device()

        # TODO make it work for more than one node
        self.spawn_kwargs["nproc_per_node"] = self.nproc_per_node
        if self.backend == "xla-tpu" and self.with_amp:
            raise RuntimeError("The value of with_amp should be False if backend is xla")

        # idist.spawn(backend, training, args=(), kwargs_dict={"config": config}, nproc_per_node=nproc_per_node)

        n_instances = len(self.env.instances)
        n_workers = self.nproc_per_node

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        all_ids = np.arange(0, n_instances)
        for ids in chunks(all_ids, n_workers):
            instance_ids = ids
            checkpoint_ids = ids
            config["instance_ids"] = instance_ids
            config["checkpoint_ids"] = checkpoint_ids
            with idist.Parallel(backend=self.backend, **self.spawn_kwargs) as parallel:
                parallel.run(self._evaluate, config)


        # while self.total_eval_runs_required > 0:
        #     for worker_id in range(self.n_workers):
        #         checkpoint_filename = self._evaluate(0, instance_id=instance_id, worker_id=worker_id)
        #         instance_id += 1
        #         if instance_id >= len(self.instances):
        #             instance_id = 0
        #             checkpoint_id += 1
        #         self.visited_checkpoint_filenames.append(checkpoint_filename)
        #         self.total_eval_runs_required -= 1
        #         if self.total_eval_runs_required <= 0:
        #             break
        #     # TODO dump results

        print("done")
        idist.finalize()  # end all subprocesses

    def _evaluate(self, local_rank, config):
        """
        Evaluation function / part which is parallelized
        """
        instance_ids = config["instance_ids"]
        checkpoint_ids = config["checkpoint_ids"]
        instance_id = instance_ids[local_rank]
        checkpoint_id = checkpoint_ids[local_rank]

        instance = self.instances[instance_id]
        checkpoint_data = self.checkpoint_data[checkpoint_id]
        checkpoint_filename = checkpoint_data['checkpoint_path']
        checkpoint = torch.load(checkpoint_filename)
        self.architecture.copy().load_state_dict(checkpoint['model'])  # TODO do we need to copy the architecture?
        # TODO do we need to check device here and move to appropriate device?
        policy = self.architecture

        evalworker = EvaluationRolloutWorker(
            policy=policy,
            policy_type=self.policy_type,
            device=self.device,
        )
        env = deepcopy(self.env)
        env.instance_set={0: instance}
        steps, rewards, instances = evalworker.eval(env, self.n_episodes_per_instance)
        assert instance == instances[0], 'Environment did not use the required instance'

        checkpoint_data['instances'] = instance
        checkpoint_data['rewards'] = rewards
        checkpoint_data['eval_steps'] = steps
        checkpoint_data['checkpoint_evaluated'] = True

        # TODO add a file lock?
        with open(self.output_file, 'a+') as out_fh:
            json.dump(checkpoint_data, out_fh)

        return checkpoint_filename



