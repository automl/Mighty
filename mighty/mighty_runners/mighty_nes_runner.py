from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from mighty.mighty_runners.mighty_runner import MightyRunner

# TODO: check if installed and exit if not
from evosax import xNES, SNES, FitnessShaper
import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MightyNESRunner(MightyRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        es_cls = xNES if cfg.es == "xnes" else SNES
        es_kwargs = {}
        if "es_kwargs" in cfg.keys():
            es_kwargs = cfg.es_kwargs
        total_n_params = sum([len(p.flatten()) for p in self.agent.parameters])
        self.es = es_cls(popsize=cfg.popsize, num_dims=total_n_params, **es_kwargs)
        self.rng = jax.random.PRNGKey(0)
        self.fit_shaper = FitnessShaper(centered_rank=True, w_decay=0.0, maximize=True)
        self.iterations = cfg.iterations

    def run(self):
        es_state = self.es.initialize(self.rng)
        for _ in range(self.iterations):
            rng_ask, _ = jax.random.split(self.rng, 2)
            x, es_state = self.es.ask(rng_ask, es_state)
            eval_rewards = []
            for individual in x:
                # 1. Make tensor from x
                individual = np.asarray(individual)
                individual = torch.tensor(individual, dtype=torch.float32)
                # 2. Shape it to match the model's parameters
                param_shapes = [p.shape for p in self.agent.parameters]
                reshaped_individual = []
                for shape in param_shapes:
                    new_individual = individual[: shape.numel()]
                    new_individual = new_individual.reshape(shape)
                    reshaped_individual.append(new_individual)
                    individual = individual[shape.numel() :]
                # 3. Set the model's parameters to the shaped tensor
                for p, x_ in zip(self.agent.parameters, reshaped_individual):
                    p.data = x_
                eval_results = self.evaluate()
                eval_rewards.append(eval_results["mean_eval_reward"])
            fitness = self.fit_shaper.apply(x, jnp.array(eval_rewards))
            es_state = self.es.tell(x, fitness, es_state)
        self.close()
        eval_results = self.evaluate()
        return {"step": self.iterations}, eval_results
