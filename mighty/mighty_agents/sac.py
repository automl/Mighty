from pathlib import Path
from typing import Optional, Dict, List, Type, Union

import numpy as np
import torch
from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class
from mighty.mighty_exploration import StochasticPolicy, MightyExplorationPolicy
from mighty.mighty_update import SACUpdate
from mighty.mighty_models.sac import SACModel
from omegaconf import DictConfig
from mighty.mighty_replay import MightyReplay
from mighty.mighty_utils.logger import Logger

from mighty.mighty_replay import TransitionBatch

class MightySACAgent(MightyAgent):
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
        replay_buffer_class: Optional[Type[MightyReplay]] = MightyReplay,
        replay_buffer_kwargs: Optional[Dict] = None,
        meta_methods: Optional[List[Type]] = None,
        meta_kwargs: Optional[List[Dict]] = None,
        n_policy_units: int = 8,
        n_critic_units: int = 8,
        soft_update_weight: float = 0.01,
        policy_class: Optional[
            Union[str, DictConfig, Type[MightyExplorationPolicy]]
        ] = None,
        policy_kwargs: Optional[Dict] = None,
        tau: float = 0.005,
        alpha: float = 0.2,
        n_gradient_steps: int = 1,
    ):
        """Initialize the SAC agent.

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
        :param replay_buffer_class: Replay buffer class
        :param replay_buffer_kwargs: Arguments for the replay buffer
        :param meta_methods: Meta methods for the agent
        :param meta_kwargs: Arguments for meta methods
        :param n_policy_units: Number of units for the policy network
        :param n_critic_units: Number of units for the critic network
        :param soft_update_weight: Size of soft updates for the target network
        :param policy_class: Policy class
        :param policy_kwargs: Arguments for the policy
        :param tau: Soft update parameter
        :param alpha: Entropy coefficient
        :param n_gradient_steps: Number of gradient steps per update
        """
        self.gamma = gamma
        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units
        self.soft_update_weight = soft_update_weight
        self.tau = tau
        self.alpha = alpha
        self.n_gradient_steps = n_gradient_steps

        # Placeholder variables which are filled in self._initialize_agent
        self.model: SACModel | None = None
        self.update_fn: SACUpdate | None = None

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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
        )

    def _initialize_agent(self):
        """Initialize SAC specific components."""

        self.model = SACModel(
            obs_size=self.env.single_observation_space.shape[0],
            action_size=self.env.single_action_space.shape[0],
        )
        self.policy = self.policy_class(
            algo=self, model=self.model, **self.policy_kwargs
        )
        self.update_fn = SACUpdate(
            model=self.model,
            policy_lr=self.learning_rate,
            q_lr=self.learning_rate,
            value_lr=self.learning_rate,
            tau=self.tau,
            alpha=self.alpha,
            gamma=self.gamma,
        )

    @property
    def value_function(self):
        """Return the value function model."""
        return self.model.value_net

    def update_agent(self, **kwargs) -> Dict[str, float]:
        """Update the agent using SAC.

        :return: Dictionary containing the update metrics.
        """
        if len(self.buffer) < self._learning_starts:
            return {}

        metrics_sac = {}

        for _ in range(self.n_gradient_steps):
            transition_batch = self.buffer.sample(batch_size=self._batch_size)
            metrics_sac.update(self.update_fn.update(transition_batch))

        # Log metrics
        self.logger.log("Update/q_loss1", metrics_sac["q_loss1"])
        self.logger.log("Update/q_loss2", metrics_sac["q_loss2"])
        self.logger.log("Update/policy_loss", metrics_sac["policy_loss"])

        return metrics_sac

    def process_transition(self, curr_s, action, reward, next_s, dones, log_prob=None, metrics=None):
        
        # convert into a transition object
        transition = TransitionBatch(curr_s, action, reward, next_s, dones)
        
        if "rollout_values" not in metrics:
            metrics["rollout_values"] = np.empty((0, self.env.single_action_space.n))

        # Add Td-error to metrics
        metrics["td_error"] = (
            self.qlearning.td_error(transition, self.q, self.q_target).detach().numpy()
        )
        
        # Compute and add rollout values to metrics
        values = (
            self.value_function(
                torch.as_tensor(transition.observations, dtype=torch.float32)
            )
            .detach()
            .numpy()
            .reshape((transition.observations.shape[0], -1))
        )

        metrics["rollout_values"] = np.append(metrics["rollout_values"], values, axis=0)
        
        # Add the transition to the buffer
        self.buffer.add(transition, metrics)
        
        return metrics

    def save(self, t: int):
        """Save current agent state."""
        super().make_checkpoint_dir(t)
        torch.save(
            self.model.policy_net.state_dict(), self.checkpoint_dir / "policy_net.pt"
        )
        torch.save(self.model.q_net1.state_dict(), self.checkpoint_dir / "q_net1.pt")
        torch.save(self.model.q_net2.state_dict(), self.checkpoint_dir / "q_net2.pt")
        torch.save(
            self.model.value_net.state_dict(), self.checkpoint_dir / "value_net.pt"
        )
        torch.save(
            self.update_fn.policy_optimizer.state_dict(),
            self.checkpoint_dir / "policy_optimizer.pt",
        )
        torch.save(
            self.update_fn.q_optimizer1.state_dict(),
            self.checkpoint_dir / "q_optimizer1.pt",
        )
        torch.save(
            self.update_fn.q_optimizer2.state_dict(),
            self.checkpoint_dir / "q_optimizer2.pt",
        )
        torch.save(
            self.update_fn.value_optimizer.state_dict(),
            self.checkpoint_dir / "value_optimizer.pt",
        )
        
        if self.verbose:
            print(f"Saved checkpoint at {self.checkpoint_dir}")

    def load(self, path: str):
        """Load the internal state of the agent."""
        base_path = Path(path)
        self.model.policy_net.load_state_dict(torch.load(base_path / "policy_net.pt"))
        self.model.q_net1.load_state_dict(torch.load(base_path / "q_net1.pt"))
        self.model.q_net2.load_state_dict(torch.load(base_path / "q_net2.pt"))
        self.model.value_net.load_state_dict(torch.load(base_path / "value_net.pt"))
        self.update_fn.policy_optimizer.load_state_dict(
            torch.load(base_path / "policy_optimizer.pt")
        )
        self.update_fn.q_optimizer1.load_state_dict(
            torch.load(base_path / "q_optimizer1.pt")
        )
        self.update_fn.q_optimizer2.load_state_dict(
            torch.load(base_path / "q_optimizer2.pt")
        )
        self.update_fn.value_optimizer.load_state_dict(
            torch.load(base_path / "value_optimizer.pt")
        )
        
        if self.verbose:
            print(f"Loaded checkpoint at {path}")

    @property
    def agent_type(self):
        return "SAC"
