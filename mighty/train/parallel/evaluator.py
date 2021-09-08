from typing import List, Optional
import torch
import torch.nn as nn

from mighty.train.parallel.rollout import EvaluationRolloutWorker


class Evaluator(object):
    def __init__(
            self,
            checkpoint_dir: str,
            device: str,
            n_workers: int = 1,
            n_episodes_per_instance: int = 1,
            instance_ids: Optional[List[int]] = None,
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
        self.instance_ids = instance_ids  # only evaluate on those instances from the training set
        self.visited_checkpoint_filenames = []  # add filenames of evaluated checkpoints here

        # TODO: read in checkpoint filenames
        # TODO: sort by date, first work on oldest
        # TODO: get agent type
        # TODO: get architecture
        self.architecture = None  # type: nn.Module
        self.policy_type = "continuous"  # TODO make function: determine policy type (discrete or continuous)
        # TODO: get all train instances
        # TODO: setup logging / logfile
        # TODO: setup idist

    def evaluate(self,):
        for checkpoint_filename in self.checkpoint_filenames:
            if checkpoint_filename in self.visited_checkpoint_filenames:
                continue
            checkpoint = torch.load(checkpoint_filename)
            self.architecture.copy().load_state_dict(checkpoint['model'])  # TODO do we need to copy the architecture?
            # TODO do we need to check device here and move to appropriate device?
            policy = self.architecture

            evalworker = EvaluationRolloutWorker(
                policy=policy,
                policy_type=self.policy_type,
                device=self.device,
            )

            # TODO create environments
            # TODO distribute eval
            # TODO gather results

            self.visited_checkpoint_filenames.append(checkpoint_filename)
        # TODO dump results



