import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class CosineLRSchedule(MightyMetaComponent):
    def __init__(self, initial_lr, num_decay_steps, min_lr, restart_every=10000, restart_multiplier=1.2) -> None:
        super().__init__()
        self.restart_every = restart_every
        self.t_mult = restart_multiplier
        self.eta_max = initial_lr
        self.t = 0
        self.t_max = num_decay_steps
        self.eta_min = min_lr
        self.pre_step_methods = [self.adapt_lr]

    def adapt_lr(self, metrics):
        self.t += 1
        if self.restart_every > 0 and self.t >= self.restart_every:
            self.t = 0
            self.eta_max = self.eta_max * self.t_mult
            metrics["hp/lr"] = self.eta_max
        else:
            metrics["hp/lr"] = self.eta_min + 0.5*(self.eta_max-self.eta_min)*(1+np.cos((self.t/self.t_max)*np.pi))