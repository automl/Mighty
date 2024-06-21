"""Curriculum Learning via Self-Paced Context Evaluation."""

from __future__ import annotations

import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class SPaCE(MightyMetaComponent):
    """Curriculum Learning via Self-Paced Context Evaluation."""

    def __init__(self, criterion="relative_improvement", threshold=0.1, k=1) -> None:
        """SPaCE initialization.

        :param criterion: Ranking criterion
        :param threshold: Minimum average change needed to keep train set size
        :param k: Size of instance set increase
        :return:
        """
        super().__init__()
        self.criterion = criterion
        self.threshold = threshold
        self.instance_set = []
        self.increase_by_k_instances = k
        self.current_instance_set_size = k
        self.last_evals = None
        self.all_instances = None
        self.pre_episode_methods = [self.get_instances]

    def get_instances(self, metrics):
        """Get Training set on episode start.

        :param metrics: Current metrics dict
        :return:
        """
        env = metrics["env"]
        vf = metrics["vf"]
        rollout_values = None
        if "rollout_values" in metrics:
            rollout_values = metrics["rollout_values"]

        if self.all_instances is None:
            self.all_instances = np.array(env.instance_id_list.copy())

        if self.last_evals is None and rollout_values is None:
            self.instance_set = np.random.default_rng().choice(
                    self.all_instances, size=self.current_instance_set_size
                )
        elif self.last_evals is None:
            self.instance_set = np.random.default_rng().choice(
                    self.all_instances, size=self.current_instance_set_size
                )
            self.last_evals = np.nanmean(rollout_values)
        else:
            if (
                abs(np.mean(rollout_values) - self.last_evals)
                / (self.last_evals + 1e-6)
                <= self.threshold
            ):
                self.current_instance_set_size += self.increase_by_k_instances
            self.last_evals = np.nanmean(rollout_values)
            evals = self.get_evals(env, vf)
            if self.criterion == "improvement":
                improvement = evals - self.last_evals
            elif self.criterion == "relative_improvement":
                improvement = (evals - self.last_evals) / self.last_evals
            else:
                raise NotImplementedError("This SpaCE criterion is not implemented.")
            self.instance_set = self.all_instances[np.argsort(improvement)[::-1]][
                : self.current_instance_set_size
            ]
        env.instance_set = self.instance_set

    def get_evals(self, env, vf):
        """Get values for s_0 of all instances.

        :param env: environment
        :param vf: value or q function
        :return:
        """
        values = []
        for i in self.all_instances:
            state, _ = env.reset()
            env.inst_ids = [i]
            v = np.array(vf(state))
            # If we're dealing with a q function, we transform to value here
            if len(v) > 1:
                v = [sum(v)]
            values.append(v[0])
        return values
