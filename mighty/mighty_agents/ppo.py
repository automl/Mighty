from pathlib import Path
from typing import Optional, Dict, List, Type, Union

import numpy as np
import torch
from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class
from mighty.mighty_exploration import StochasticPolicy, MightyExplorationPolicy
from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_update.ppo_update import PPOUpdate
from mighty.mighty_replay.mighty_rollout_buffer import MightyRolloutBuffer
from mighty.mighty_utils.logger import Logger

from omegaconf import DictConfig


class MightyPPOAgent(MightyAgent):
    def __init__(
        self,
        env,
        logger: Logger,
        eval_env=None,
        seed: Optional[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: Optional[Dict] = None,
        rollout_buffer_class: Optional[Type[MightyRolloutBuffer]] = MightyRolloutBuffer,
        rollout_buffer_kwargs: Optional[Dict] = None,
        meta_methods: Optional[List[Type]] = None,
        meta_kwargs: Optional[List[Dict]] = None,
        n_policy_units: int = 8,
        n_critic_units: int = 8,
        soft_update_weight: float = 0.01,
        policy_class: Optional[
            Union[str, DictConfig, Type[MightyExplorationPolicy]]
        ] = None,
        policy_kwargs: Optional[Dict] = None,
        ppo_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_gradient_steps: int = 10,
    ):
        # FIXME: missing docstring
        self.gamma = gamma
        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units
        self.soft_update_weight = soft_update_weight
        self.ppo_clip = ppo_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_gradient_steps = n_gradient_steps

        # Placeholder variables which are filled in self._initialize_agent
        self.model: PPOModel | None = None
        self.update_fn: PPOUpdate | None = None

        # Policy class
        policy_class = retrieve_class(cls=policy_class, default_cls=StochasticPolicy)
        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

        super().__init__(
            env=env,
            logger=logger,
            seed=seed,
            eval_env=eval_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            learning_starts=learning_starts,
            render_progress=render_progress,
            log_tensorboard=log_tensorboard,
            log_wandb=log_wandb,
            wandb_kwargs=wandb_kwargs,
            replay_buffer_class=rollout_buffer_class,  # Use rollout buffer
            replay_buffer_kwargs=rollout_buffer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
        )

    def _initialize_agent(self):
        """Initialize PPO specific components."""
        self.model = PPOModel(
            obs_size=self.env.single_observation_space.shape[0],
            action_size=self.env.single_action_space.n,
        )
        self.policy = self.policy_class(
            algo=self, model=self.model, **self.policy_kwargs
        )
        self.update_fn = PPOUpdate(
            model=self.model,
            policy_lr=self.learning_rate,
            value_lr=self.learning_rate,
            epsilon=self.ppo_clip,
            ent_coef=self.entropy_coef,
            vf_coef=self.value_loss_coef,
            max_grad_norm=self.max_grad_norm,
        )

    @property
    def value_function(self):
        """Return the value function model."""
        return self.model.value_net

    def update_agent(self) -> Dict[str, float]:
        if len(self.buffer) < self._learning_starts:
            return {}

        metrics = {}
        for _ in range(self.n_gradient_steps):
            for batch in self.buffer.sample(self._batch_size):
                metrics.update(self.update_fn.update(batch))
        return metrics

    def get_transition_metrics(
        self, transition, metrics: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if "rollout_values" not in metrics:
            metrics["rollout_values"] = np.empty((0,))

        # FIXME: they were here all along?! Why did you compute them again in the base agent?
        values = (
            self.value_function(
                torch.as_tensor(transition["observations"], dtype=torch.float32)
            )
            .detach()
            .numpy()
            .reshape((transition["observations"].shape[0],))
        )

        metrics["rollout_values"] = np.append(metrics["rollout_values"], values, axis=0)

        return metrics

    # FIXME: both of these don't have any log messages, we should add them for a verbose mode
    def save(self, t: int):
        """Save current agent state."""
        super().make_checkpoint_dir(t)
        torch.save(
            self.model.policy_net.state_dict(), self.checkpoint_dir / "policy_net.pt"
        )
        torch.save(
            self.model.value_net.state_dict(), self.checkpoint_dir / "value_net.pt"
        )
        torch.save(
            self.update_fn.policy_optimizer.state_dict(),
            self.checkpoint_dir / "policy_optimizer.pt",
        )
        torch.save(
            self.update_fn.value_optimizer.state_dict(),
            self.checkpoint_dir / "value_optimizer.pt",
        )

    def load(self, path: str):
        """Load the internal state of the agent."""
        base_path = Path(path)
        self.model.policy_net.load_state_dict(torch.load(base_path / "policy_net.pt"))
        self.model.value_net.load_state_dict(torch.load(base_path / "value_net.pt"))
        self.update_fn.policy_optimizer.load_state_dict(
            torch.load(base_path / "policy_optimizer.pt")
        )
        self.update_fn.value_optimizer.load_state_dict(
            torch.load(base_path / "value_optimizer.pt")
        )

    @property
    def agent_type(self):
        return "PPO"
