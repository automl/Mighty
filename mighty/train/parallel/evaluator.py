import os
from typing import List, Optional
import json
from copy import deepcopy

import torch
import torch.nn as nn

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
            watch_mode: bool = False  # watches if new checkpoints are arriving, should we do this or later?
    ):
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

        self.checkpoint_dir

    def evaluate(self):
        instance_id = 0
        checkpoint_id = 0
        # TODO this allows to run multiple workers if we have a non-blocking behaviour. No clue how it actually is with ignite
        while self.total_eval_runs_required > 0:
            for worker_id in range(self.n_workers):
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

                with open(self.output_file, 'a+') as out_fh:
                    json.dump(checkpoint_data, out_fh)
                instance_id += 1
                if instance_id >= len(self.instances):
                    instance_id = 0
                    checkpoint_id += 1
                self.visited_checkpoint_filenames.append(checkpoint_filename)
                self.total_eval_runs_required -= 1
                if self.total_eval_runs_required <= 0:
                    break
            # TODO dump results



