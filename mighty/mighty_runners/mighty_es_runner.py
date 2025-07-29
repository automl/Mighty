from __future__ import annotations

import importlib.util as iutil
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch

# FIXME: This is a hack around our current JAX version not having these functions.
# Remove this once we upgrade JAX to a version that has these functions.
try:
    import scipy.linalg

    # Patch missing functions that JAX expects
    if not hasattr(scipy.linalg, "tril"):
        scipy.linalg.tril = np.tril
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
    if not hasattr(scipy.linalg, "tri"):
        scipy.linalg.tri = np.tri
except ImportError:
    pass

from mighty.mighty_agents.base_agent import retrieve_class
from mighty.mighty_runners.mighty_runner import MightyRunner

spec = iutil.find_spec("evosax")
found = spec is not None
if found:
    import jax
    from evosax import FitnessShaper, xNES  # type: ignore
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

        # Store original values for restoration if needed
        self.original_values = {}

        if "parameters" in self.search_targets:
            self.search_params = True
            self.total_n_params = sum([len(p.flatten()) for p in self.agent.parameters])
            num_dims -= 1  # Remove "parameters" from count
            num_dims += self.total_n_params  # Add actual parameter count

            # Store original parameters
            self.original_values["parameters"] = [
                p.clone() for p in self.agent.parameters
            ]

        # Store original values for other search targets
        for target in self.search_targets:
            if target != "parameters":
                self.original_values[target] = getattr(self.agent, target)

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

    def apply_parameters(self, individual_params) -> None:
        """Apply parameter values to the agent's model parameters."""
        individual_params = np.asarray(individual_params)
        individual_params = torch.tensor(individual_params, dtype=torch.float32)

        # Shape it to match the model's parameters
        param_shapes = [p.shape for p in self.agent.parameters]
        reshaped_individual = []
        start_idx = 0

        for shape in param_shapes:
            num_elements = shape.numel()
            end_idx = start_idx + num_elements
            new_individual = individual_params[start_idx:end_idx]
            new_individual = new_individual.reshape(shape)
            reshaped_individual.append(new_individual)
            start_idx = end_idx

        # Set the model's parameters to the shaped tensor
        for p, x_ in zip(self.agent.parameters, reshaped_individual):
            p.data = x_.clone()

    def apply_individual(self, individual) -> None:
        """Apply an individual's values to both parameters and other search targets."""
        individual = np.asarray(individual)
        current_idx = 0

        # Apply parameters if they're being searched
        if self.search_params:
            param_values = individual[current_idx : current_idx + self.total_n_params]
            self.apply_parameters(param_values)
            current_idx += self.total_n_params

        # Apply other search targets
        for target in self.search_targets:
            if target == "parameters":
                continue  # Already handled above

            new_value = individual[current_idx]

            # Handle integer-type parameters
            if target in ["_batch_size", "n_units"]:
                new_value = max(1, int(round(new_value)))  # Ensure at least 1

            setattr(self.agent, target, new_value)
            current_idx += 1

    def restore_original_values(self) -> None:
        """Restore original values before applying new individual."""
        if self.search_params:
            for p, orig_p in zip(
                self.agent.parameters, self.original_values["parameters"]
            ):
                p.data = orig_p.clone()

        for target in self.search_targets:
            if target != "parameters":
                setattr(self.agent, target, self.original_values[target])

    def run(self) -> Tuple[Dict, Dict]:
        es_state = self.es.initialize(self.rng)

        for iteration in range(self.iterations):
            rng_ask, self.rng = jax.random.split(self.rng, 2)
            x, es_state = self.es.ask(rng_ask, es_state)
            eval_rewards = []

            for individual in x:
                # Restore original values before applying new individual
                self.restore_original_values()

                # Apply the individual's values
                self.apply_individual(individual)

                # Train the agent if specified
                if self.train_agent:
                    self.train(self.num_steps_per_iteration)

                # Evaluate and collect reward
                eval_results = self.evaluate()
                eval_rewards.append(eval_results["mean_eval_reward"])

            # Update ES state with fitness
            fitness = self.fit_shaper.apply(x, jnp.array(eval_rewards))
            es_state = self.es.tell(x, fitness, es_state)

        # Final evaluation
        eval_results = self.evaluate()
        return {"step": self.iterations}, eval_results
