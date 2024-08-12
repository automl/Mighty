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
from mighty.mighty_replay import RolloutBatch
from mighty.mighty_utils.env_handling import MIGHTYENV
from omegaconf import DictConfig
from mighty.mighty_utils.types import TypeKwargs


class MightyPPOAgent(MightyAgent):
    def __init__(
        self,
        env: MIGHTYENV,  # type: ignore
        logger: Logger,
        eval_env: Optional[MIGHTYENV] = None,  # type: ignore
        seed: Optional[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: Optional[Dict] = None,
        rollout_buffer_class: Optional[
            str | DictConfig | Type[MightyRolloutBuffer]
        ] = MightyRolloutBuffer,
        rollout_buffer_kwargs: Optional[TypeKwargs] = None,
        meta_methods: Optional[List[str | type]] = None,
        meta_kwargs: Optional[List[TypeKwargs]] = None,
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
        """Initialize the PPO agent.

        Creates all relevant class variables and calls the agent-specific init function.

        :param env: Train environment
        :param logger: Mighty logger
        :param eval_env: Evaluation environment
        :param seed: Seed for random number generators
        :param learning_rate: Learning rate for training
        :param gamma: Discount factor
        :param batch_size: Batch size for training
        :param learning_starts: Number of steps before learning starts
        :param render_progress: Whether to render progress
        :param log_tensorboard: Log to TensorBoard as well as to file
        :param log_wandb: Log to Weights and Biases
        :param wandb_kwargs: Arguments for Weights and Biases logging
        :param rollout_buffer_class: Rollout buffer class
        :param rollout_buffer_kwargs: Arguments for the rollout buffer
        :param meta_methods: Meta methods for the agent
        :param meta_kwargs: Arguments for meta methods
        :param n_policy_units: Number of units for the policy network
        :param n_critic_units: Number of units for the critic network
        :param soft_update_weight: Size of soft updates for the target network
        :param policy_class: Policy class
        :param policy_kwargs: Arguments for the policy
        :param ppo_clip: Clipping parameter for PPO
        :param value_loss_coef: Coefficient for the value loss
        :param entropy_coef: Coefficient for the entropy loss
        :param max_grad_norm: Maximum gradient norm
        :param n_gradient_steps: Number of gradient steps per update
        """

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
        policy_class = retrieve_class(cls=policy_class, default_cls=StochasticPolicy)  # type: ignore
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
            replay_buffer_class=rollout_buffer_class,
            replay_buffer_kwargs=rollout_buffer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
        )

    def _initialize_agent(self) -> None:
        """Initialize PPO specific components."""

        self.buffer_kwargs["buffer_size"] = self._batch_size  # type: ignore
        self.buffer_kwargs["obs_shape"] = self.env.single_observation_space.shape[0]  # type: ignore

        if self.env.single_action_space.__class__.__name__ == "Discrete":  # type: ignore
            self.buffer_kwargs["act_dim"] = int(self.env.single_action_space.n)  # type: ignore
            self.discrete_action = True
        else:
            self.buffer_kwargs["act_dim"] = int(self.env.single_action_space.shape[0])  # type: ignore
            self.discrete_action = False

        self.buffer_kwargs["n_envs"] = self.env.observation_space.shape[0]  # type: ignore

        self.model = PPOModel(
            obs_size=self.env.single_observation_space.shape[0],  # type: ignore
            action_size=(
                self.env.single_action_space.n  # type: ignore
                if self.discrete_action
                else self.env.single_action_space.shape[0]  # type: ignore
            ),
            continuous_action=not self.discrete_action,
        )
        self.policy = self.policy_class(
            algo=self,
            model=self.model,
            discrete=self.discrete_action,
            **self.policy_kwargs,
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
    def value_function(self) -> torch.nn.Module:
        """Return the value function model."""
        return self.model.value_net  # type: ignore

    def update_agent(self, next_s, dones, **kwargs) -> Dict:  # type: ignore
        """Update the agent using PPO.

        :return: Dictionary containing the update metrics.
        """
        if len(self.buffer) < self._learning_starts:  # type: ignore
            return {}

        # Compute returns and advantages for PPO
        last_values = self.value_function(
            torch.as_tensor(next_s, dtype=torch.float32)
        ).detach()

        self.buffer.compute_returns_and_advantage(last_values, dones)  # type: ignore

        metrics: Dict = {}
        for _ in range(self.n_gradient_steps):
            for batch in self.buffer.sample(self._batch_size):  # type: ignore
                metrics.update(self.update_fn.update(batch))  # type: ignore

        self.buffer.reset()  # type: ignore

        return metrics

    def process_transition(  # type: ignore
        self,
        curr_s,
        action,
        reward,
        next_s,
        dones,
        log_prob=None,
        metrics=None,
    ) -> Dict:
        values = (
            self.value_function(torch.as_tensor(curr_s, dtype=torch.float32))
            .detach()
            .numpy()
            .reshape((curr_s.shape[0],))
        )

        rollout_batch = RolloutBatch(
            observations=curr_s,
            actions=action,
            rewards=reward,
            advantages=np.zeros_like(reward),  # Placeholder, compute later
            returns=np.zeros_like(reward),  # Placeholder, compute later
            episode_starts=dones,
            log_probs=log_prob,
            values=values,
        )

        self.buffer.add(rollout_batch, metrics)  # type: ignore

        return metrics  # type: ignore

    def save(self, t: int) -> None:
        """Save current agent state."""
        super().make_checkpoint_dir(t)
        torch.save(
            self.model.policy_net.state_dict(),  # type: ignore
            self.checkpoint_dir / "policy_net.pt",
        )
        torch.save(
            self.model.value_net.state_dict(),  # type: ignore
            self.checkpoint_dir / "value_net.pt",
        )
        torch.save(
            self.update_fn.policy_optimizer.state_dict(),  # type: ignore
            self.checkpoint_dir / "policy_optimizer.pt",
        )
        torch.save(
            self.update_fn.value_optimizer.state_dict(),  # type: ignore
            self.checkpoint_dir / "value_optimizer.pt",
        )

        if self.verbose:
            print(f"Saved checkpoint at {self.checkpoint_dir}")

    def load(self, path: str) -> None:
        """Load the internal state of the agent."""
        base_path = Path(path)
        self.model.policy_net.load_state_dict(torch.load(base_path / "policy_net.pt"))  # type: ignore
        self.model.value_net.load_state_dict(torch.load(base_path / "value_net.pt"))  # type: ignore
        self.update_fn.policy_optimizer.load_state_dict(  # type: ignore
            torch.load(base_path / "policy_optimizer.pt")
        )
        self.update_fn.value_optimizer.load_state_dict(  # type: ignore
            torch.load(base_path / "value_optimizer.pt")
        )

        if self.verbose:
            print(f"Loaded checkpoint at {path}")

    @property
    def agent_type(self) -> str:
        """Return the type of the agent."""
        return "PPO"
