from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from mighty.mighty_runners.mighty_runner import MightyRunner
from mighty.mighty_agents.base_agent import retrieve_class

# TODO: check if installed and exit if not
import importlib.util as iutil
spec = iutil.find_spec("evosax")
found = spec is not None
if found:
    from evosax import FitnessShaper, xNES
    import jax
    from jax import numpy as jnp
else:
    import warnings
    warnings.warn("evosax not found, to use NES runners please install mighty[es].")

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MightyESRunner(MightyRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.search_targets = cfg.search_targets
        num_dims = len(self.search_targets)
        self.search_params = False
        if "parameters" in self.search_targets:
            self.search_params = True
            self.total_n_params = sum([len(p.flatten()) for p in self.agent.parameters])
            num_dims -= 1
            num_dims += self.total_n_params
        
        es_cls = retrieve_class(cfg.es, default_cls=xNES)
        es_kwargs = {}
        if "es_kwargs" in cfg.keys():
            es_kwargs = cfg.es_kwargs
        
        self.es = es_cls(popsize=cfg.popsize, num_dims=num_dims, **es_kwargs)
        self.rng = jax.random.PRNGKey(0)
        self.fit_shaper = FitnessShaper(centered_rank=True, w_decay=0.0, maximize=True)
        self.iterations = cfg.iterations
        self.train_agent = cfg.rl_train_agent
        if self.train_agent:
            self.num_steps_per_iteration = cfg.num_steps_per_iteration

    def apply_parameters(self, individual):
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

    def run(self):
        es_state = self.es.initialize(self.rng)
        for _ in range(self.iterations):
            rng_ask, _ = jax.random.split(self.rng, 2)
            x, es_state = self.es.ask(rng_ask, es_state)
            eval_rewards = []
            for individual in x:
                if self.search_params:
                    self.apply_parameters(individual[:self.total_n_params])
                    individual = individual[self.total_n_params:]
                for i, target in enumerate(self.search_targets):
                    if target == "parameters":
                        continue
                    new_value = np.asarray(individual[i]).item()
                    # TODO: check for other int-only hps
                    if target in ["_batch_size", "n_units"]:
                        new_value = max(0, int(new_value))
                    setattr(self.agent, target, new_value)
                if self.train_agent:
                    self.train(self.num_steps_per_iteration)
                eval_results = self.evaluate()
                eval_rewards.append(eval_results["mean_eval_reward"])
            fitness = self.fit_shaper.apply(x, jnp.array(eval_rewards))
            es_state = self.es.tell(x, fitness, es_state)
        self.close()
        eval_results = self.evaluate()
        return {"step": self.iterations}, eval_results
