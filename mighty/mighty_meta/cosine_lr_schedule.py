"""Cosine LR Schedule with optional warm restarts."""

from __future__ import annotations

import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class CosineLRSchedule(MightyMetaComponent):
    """Cosine LR Schedule with optional warm restarts."""

    def __init__(
        self,
        initial_lr,
        num_decay_steps,
        min_lr=0,
        restart_every=10000,
        restart_multiplier=1.2,
    ) -> None:
        """Cosine schedule initialization.

        :param initial_lr: Initial maximal LR
        :param num_decay_steps: Length of schedule in steps
        :param min_lr: Minimal LR
        :param restart_every: Restart frequency
        :param restart multiplier: Multiplies current learning rate on restart.
        :return:
        """
        super().__init__()
        self.restart_every = restart_every
        self.n_restarts = 0
        self.t_mult = restart_multiplier
        self.eta_max = initial_lr
        self.t_max = num_decay_steps
        self.eta_min = min_lr
        self.pre_step_methods = [self.adapt_lr]

    def adapt_lr(self, metrics):
        """Adapt LR on step.

        :param metrics: Dict of current metrics
        :return:
        """
        reset = False
        if self.restart_every > 0:
            if self.n_restarts < np.floor(metrics["step"] / self.restart_every):
                self.n_restarts += 1
                self.eta_max = (
                    self.eta_min
                    + 0.5
                    * (self.eta_max - self.eta_min)
                    * (1 + np.cos((metrics["step"] / self.t_max) * np.pi))
                    * self.t_mult
                )
                metrics["hp/lr"] = self.eta_max
                reset = True

        if metrics["step"] < self.t_max and not reset:
            metrics["hp/lr"] = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
                1 + np.cos((metrics["step"] / self.t_max) * np.pi)
            )
