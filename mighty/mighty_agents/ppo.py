from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch
from omegaconf import DictConfig

from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class
from mighty.mighty_exploration import MightyExplorationPolicy, StochasticPolicy
from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_replay import RolloutBatch
from mighty.mighty_replay.mighty_rollout_buffer import MightyRolloutBuffer
from mighty.mighty_update.ppo_update import PPOUpdate
from mighty.mighty_utils.mighty_types import MIGHTYENV, TypeKwargs


class MightyPPOAgent(MightyAgent):
    def __init__(
        self,
        output_dir,
        env: MIGHTYENV,  # type: ignore
        eval_env: Optional[MIGHTYENV] = None,  # type: ignore
        seed: Optional[int] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_wandb: bool = False,
        wandb_kwargs: dict | None = None,
        log_infos: bool = False,
        rollout_buffer_class: Optional[
            str | DictConfig | Type[MightyRolloutBuffer]
        ] = MightyRolloutBuffer,
        rollout_buffer_kwargs: Optional[TypeKwargs] = {
            "buffer_size": 256,
        },
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
        hidden_sizes: Optional[List[int]] = [64, 64],
        activation: Optional[str] = "tanh",
        n_epochs: int = 10,
        minibatch_size: int = 32,
        kl_target: float = 0.001,
        use_value_clip: bool = True,
        value_clip_eps: float = 0.2,
        total_timesteps: int = 1_000_000,
        normalize_obs: bool = False,
        normalize_reward: bool = False,
        rescale_action: bool = False,
        tanh_squash: bool = False,
    ):
        """Initialize the PPO agent.

        Creates all relevant class variables and calls the agent-specific init function.

        :param env: Train environment
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

        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.n_policy_units = n_policy_units
        self.n_critic_units = n_critic_units
        self.soft_update_weight = soft_update_weight
        self.ppo_clip = ppo_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.kl_target = kl_target
        self.use_value_clip = use_value_clip
        self.value_clip_eps = value_clip_eps
        self.tanh_squash = tanh_squash

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
            output_dir=output_dir,
            seed=seed,
            eval_env=eval_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            learning_starts=learning_starts,
            n_gradient_steps=n_gradient_steps,
            render_progress=render_progress,
            log_wandb=log_wandb,
            wandb_kwargs=wandb_kwargs,
            replay_buffer_class=rollout_buffer_class,
            replay_buffer_kwargs=rollout_buffer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            rescale_action=rescale_action,
            log_infos=log_infos,
        )

        self.loss_buffer = {
            "Update/policy_loss": [],
            "Update/value_loss": [],
            "Update/entropy": [],
            "Update/approx_kl": [],
            "update_at_step": [],
        }

    def _initialize_agent(self) -> None:
        """Initialize PPO specific components."""

        self.buffer_kwargs["obs_shape"] = self.env.single_observation_space.shape[0]  # type: ignore

        if self.env.single_action_space.__class__.__name__ == "Discrete":  # type: ignore
            self.buffer_kwargs["act_dim"] = int(self.env.single_action_space.n)  # type: ignore
            self.discrete_action = True
        else:
            self.buffer_kwargs["act_dim"] = int(self.env.single_action_space.shape[0])  # type: ignore
            self.discrete_action = False

        self.buffer_kwargs["n_envs"] = self.env.num_envs  # type: ignore
        self.buffer_kwargs["discrete_action"] = self.discrete_action  # type: ignore

        self.model = PPOModel(
            obs_shape=self.env.single_observation_space.shape[0],  # type: ignore
            action_size=(
                self.env.single_action_space.n  # type: ignore
                if self.discrete_action
                else self.env.single_action_space.shape[0]  # type: ignore
            ),
            continuous_action=not self.discrete_action,
            tanh_squash=self.tanh_squash,
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
            n_epochs=self.n_epochs,
            minibatch_size=self.minibatch_size,
            kl_target=self.kl_target,
            use_value_clip=self.use_value_clip,
            value_clip_eps=self.value_clip_eps,
            total_timesteps=self.total_timesteps,
        )

    @property
    def parameters(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters (policy + value) for PPO."""
        # note: self.model is a PPOModel with .policy_head and .value_head
        return list(self.model.policy_head.parameters()) + list(
            self.model.value_head.parameters()
        )

    @property
    def value_function(self) -> torch.nn.Module:
        """Return the value function model."""
        return self.model.value_function_module  # type: ignore

    def update_agent(
        self, transition_batch, batches_left, next_s, dones, **kwargs
    ) -> Dict:  # type: ignore
        """Update the agent using PPO.

        :return: Dictionary containing the update metrics.
        """
        metrics: Dict = {}
        metrics.update(self.update_fn.update(transition_batch))  # type: ignore

        for key, value in metrics.items():
            self.loss_buffer[key].append(value)
        self.loss_buffer["update_at_step"].append(self.steps)

        if batches_left == 0:
            self.buffer.reset()  # type: ignore

        return metrics

    def update(self, metrics: Dict, update_kwargs: Dict) -> Dict:
        if len(self.buffer) < self._learning_starts:  # type: ignore
            return {}

        # Compute returns and advantages for PPO
        last_values = self.value_function(
            torch.as_tensor(update_kwargs["next_s"], dtype=torch.float32)
        ).detach()

        self.buffer.compute_returns_and_advantage(last_values, update_kwargs["dones"])  # type: ignore
        if "rollout_values" in metrics:
            del metrics["rollout_values"]
            metrics["rollout_values"] = []

        if "rollout_logits" in metrics:
            del metrics["rollout_logits"]
            metrics["rollout_logits"] = []

        # FIXME: Hack Temporarily override batch_size for PPO minibatching
        original_batch_size = self._batch_size
        self._batch_size = self.minibatch_size
        result = super().update(metrics, update_kwargs)  # type: ignore
        self._batch_size = original_batch_size  # Restore original
        return result

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

        latents = (
            np.arctanh(np.clip(action, -0.999, 0.999))
            if (
                getattr(self.model, "continuous_action", False)
                and getattr(self.model, "tanh_squash", False)
            )
            else None
        )

        if log_prob is not None and log_prob.shape[-1] == 1:
            log_prob = log_prob.squeeze(-1)  # (64, 1) → (64,)

        # `dones` only carries terminations; time-limit truncations must also cut
        # the GAE trajectory, and their reward is bootstrapped with V(final_obs)
        # (next_s holds the real final observation on truncation).
        episode_starts = dones
        if metrics is not None and "transition" in metrics:
            truncated = np.asarray(metrics["transition"]["truncated"])
            episode_starts = np.logical_or(dones, truncated).astype(np.float32)
            if truncated.any():
                next_values = (
                    self.value_function(torch.as_tensor(next_s, dtype=torch.float32))
                    .detach()
                    .numpy()
                    .reshape((next_s.shape[0],))
                )
                reward = reward + self.gamma * next_values * truncated

        rollout_batch = RolloutBatch(
            observations=curr_s,
            actions=action,
            rewards=reward,
            advantages=np.zeros_like(reward),  # Placeholder, compute later
            returns=np.zeros_like(reward),  # Placeholder, compute later
            episode_starts=episode_starts,
            log_probs=log_prob,
            values=values,
            latents=latents,
        )

        self.buffer.add(rollout_batch, metrics)  # type: ignore

        if "rollout_values" not in metrics:
            metrics["rollout_values"] = np.array([])

        if "rollout_logits" not in metrics:
            metrics["rollout_logits"] = np.array([])

        metrics["rollout_values"] = np.append(metrics["rollout_values"], values)
        metrics["rollout_logits"] = np.append(metrics["rollout_logits"], values)

        return metrics  # type: ignore

    def save(self, t: int) -> None:
        """Save current agent state."""
        super().make_checkpoint_dir(t)
        torch.save(
            self.model.policy_head.state_dict(),  # type: ignore
            self.checkpoint_dir / "policy_head.pt",
        )
        torch.save(
            self.model.value_head.state_dict(),  # type: ignore
            self.checkpoint_dir / "value_head.pt",
        )
        torch.save(
            self.update_fn.optimizer.state_dict(),  # type: ignore
            self.checkpoint_dir / "optimizer.pt",
        )

        if self.verbose:
            print(f"Saved checkpoint at {self.checkpoint_dir}")

    def load(self, path: str) -> None:
        """Load the internal state of the agent."""
        base_path = Path(path)
        self.model.policy_head.load_state_dict(torch.load(base_path / "policy_head.pt"))  # type: ignore
        self.model.value_head.load_state_dict(torch.load(base_path / "value_head.pt"))  # type: ignore
        self.update_fn.optimizer.load_state_dict(  # type: ignore
            torch.load(base_path / "optimizer.pt")
        )

        if self.verbose:
            print(f"Loaded checkpoint at {path}")
