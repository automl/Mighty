"""DQN agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dill
import numpy as np
import torch
from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class
from mighty.mighty_exploration import EpsilonGreedy, MightyExplorationPolicy
from mighty.mighty_models import DQN
from mighty.mighty_update import DoubleQLearning, QLearning
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from mighty.mighty_replay import MightyReplay
    from mighty.utils.logger import Logger
    from mighty.utils.types import TypeKwargs
    from omegaconf import DictConfig

    from bbf_e.mighty.utils.env_handling import MIGHTYENV


class MightyDQNAgent(MightyAgent):
    """Mighty DQN agent.

    This agent implements the DQN algorithm and extension as first proposed in
    "Playing Atari with Deep Reinforcement Learning" by Mnih et al. in 2013.
    DDQN was proposed by van Hasselt et al. in 2016's
    "Deep Reinforcement Learning with Double Q-learning".
    Like all Mighty agents, it's supposed to be called via the train method.
    By default, this agent uses an epsilon-greedy policy.
    """

    def __init__(
        self,
        # MightyAgent Args
        env: MIGHTYENV,
        logger: Logger,
        seed: int | None = None,
        eval_env: MIGHTYENV = None,
        learning_rate: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        batch_size: int = 64,
        learning_starts: int = 1,
        render_progress: bool = True,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        wandb_kwargs: dict | None = None,
        replay_buffer_class: str | DictConfig | type[MightyReplay] | None = None,
        replay_buffer_kwargs: TypeKwargs | None = None,
        meta_methods: list[str | type] | None = None,
        meta_kwargs: list[TypeKwargs] | None = None,
        # DDQN Specific Args
        use_target: bool = True,
        n_units: int = 8,
        soft_update_weight: float = 0.01,  # TODO which default value?
        policy_class: str | DictConfig | type[MightyExplorationPolicy] | None = None,
        policy_kwargs: TypeKwargs | None = None,
        q_class: str | DictConfig | type[DQN] | None = None,
        q_kwargs: TypeKwargs | None = None,
        td_update_class: QLearning = QLearning,
        td_update_kwargs: TypeKwargs | None = None,
        save_replay: bool = False,
    ):
        """DQN initialization.

        Creates all relevant class variables and calls agent-specific init function

        :param env: Train environment
        :param logger: Mighty logger
        :param eval_env: Evaluation environment
        :param learning_rate: Learning rate for training
        :param epsilon: Exploration factor for training
        :param batch_size: Batch size for training
        :param render_progress: Render progress
        :param log_tensorboard: Log to tensorboard as well as to file
        :param replay_buffer_class: Replay buffer class from coax replay buffers
        :param replay_buffer_kwargs: Arguments for the replay buffer
        :param tracer_class: Reward tracing class from coax tracers
        :param tracer_kwargs: Arguments for the reward tracer
        :param n_units: Number of units for Q network
        :param soft_update_weight: Size of soft updates for target network
        :param policy_class: Policy class from coax value-based policies
        :param policy_kwargs: Arguments for the policy
        :param td_update_class: Kind of TD update used from coax TD updates
        :param td_update_kwargs: Arguments for the TD update
        :return:
        """
        if meta_kwargs is None:
            meta_kwargs = []
        if meta_methods is None:
            meta_methods = []
        if wandb_kwargs is None:
            wandb_kwargs = {}
        self.n_units = n_units
        assert 0.0 <= soft_update_weight <= 1.0  # noqa: PLR2004
        self.soft_update_weight = soft_update_weight

        # Placeholder variables which are filled in self.initialize_agent
        self.q: DQN | None = None
        self.policy: MightyExplorationPolicy | None = None
        self.q_target: DQN | None = None
        self.qlearning: QLearning | None = None
        self.use_target = use_target

        # Q-function Class
        q_class = retrieve_class(cls=q_class, default_cls=DQN)
        if q_kwargs is None:
            q_kwargs = {"n_layers": 0}
        self.q_class = q_class
        self.q_kwargs = q_kwargs

        # Policy Class
        policy_class = retrieve_class(cls=policy_class, default_cls=EpsilonGreedy)
        if policy_kwargs is None:
            policy_kwargs = {"epsilon": 0.1}
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

        self.td_update_class = retrieve_class(
            cls=td_update_class, default_cls=DoubleQLearning
        )
        if td_update_kwargs is None:
            td_update_kwargs = {"gamma": gamma}
        self.td_update_kwargs = td_update_kwargs
        self.save_replay = save_replay

        super().__init__(
            env=env,
            logger=logger,
            seed=seed,
            eval_env=eval_env,
            learning_rate=learning_rate,
            epsilon=epsilon,
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

    @property
    def value_function(self):
        """Q-function."""
        return self.q

    def _initialize_agent(self):
        """Initialize DQN specific things like q-function."""
        if not isinstance(self.q_kwargs, dict):
            self.q_kwargs = OmegaConf.to_container(self.q_kwargs)

        self.q = self.q_class(
            num_actions=self.env.single_action_space.n,
            obs_size=self.env.single_observation_space.shape,
            **self.q_kwargs,
        )
        self.policy = self.policy_class(algo="q", model=self.q, **self.policy_kwargs)

        # target network
        if not self.use_target:
            self.q_target = None
        else:
            q_state = self.q.state_dict()
            self.q_target = self.q_class(
                num_actions=self.env.single_action_space.n,
                obs_size=self.env.single_observation_space.shape,
                **self.q_kwargs,
            )
            self.q_target.load_state_dict(q_state)

        # specify how to update value function
        self.qlearning = self.td_update_class(model=self.q, **self.td_update_kwargs)

        print("Initialized agent.")

    def update_agent(self):
        """Compute and apply TD update.

        :param step: Current training step
        :return:
        """
        transition_batch = self.replay_buffer.sample(batch_size=self._batch_size)
        preds, targets = self.qlearning.get_targets(
            transition_batch, self.q, self.q_target
        )

        metrics_q = self.qlearning.apply_update(preds, targets)
        metrics_q["Q-Update/td_targets"] = targets.detach().numpy()
        metrics_q["Q-Update/td_errors"] = (targets - preds).detach().numpy()
        self.logger.log(
            "batch_predictions", preds.mean(axis=1).detach().numpy().tolist()
        )
        self.logger.log("td_error", metrics_q["Q-Update/td_errors"].mean().item())
        self.logger.log("loss", metrics_q["Q-Update/loss"])

        # sync target model
        if self.q_target is not None:
            for param, target_param in zip(
                self.q.parameters(), self.q_target.parameters(), strict=False
            ):
                target_param.data.copy_(
                    self.soft_update_weight * param.data
                    + (1 - self.soft_update_weight) * target_param.data
                )
        return metrics_q

    def get_transition_metrics(self, transition, metrics):
        """Get metrics per transition.

        :param transition: Current transition
        :param metrics: Current metrics dict
        :return:
        """
        if "rollout_errors" not in metrics:
            metrics["rollout_values"] = np.empty(0)

        metrics["td_error"] = (
            self.qlearning.td_error(transition, self.q, self.q_target).detach().numpy()
        )
        metrics["rollout_values"] = np.append(
            metrics["rollout_values"],
            self.value_function(
                torch.as_tensor(transition.observations, dtype=torch.float32)
            )
            .detach()
            .numpy(),
        )
        return metrics

    def save(self, t):
        """Return current agent state, e.g. for saving.

        For DQN, this consists of:
        - the Q network parameters
        - the Q network function state
        - the target network parameters
        - the target network function state

        :return: Agent state
        """
        super().make_checkpoint_dir(t)
        # Save q parameters
        q_path = self.checkpoint_dir / "q.pt"
        torch.save(self.q.state_dict(), q_path)

        # Save target parameters
        if self.q_target is not None:
            target_path = self.checkpoint_dir / "q_target.pt"
            torch.save(self.q_target.state_dict(), target_path)

        # Save optimizer state
        optimizer_path = self.checkpoint_dir / "optimizer.pkl"
        torch.save(
            {"optimizer_state": self.qlearning.optimizer.state_dict()}, optimizer_path
        )

        # Save replay buffer
        if self.save_replay:
            replay_path = self.checkpoint_dir / "replay.pkl"
            self.replay_buffer.save(replay_path)
        print(f"Saved checkpoint at {self.checkpoint_dir}")

    def load(self, path):
        """Set the internal state of the agent, e.g. after loading."""
        base_path = Path(path)
        q_path = base_path / "q.pt"
        q_state = torch.load(q_path)
        self.q.load_state_dict(q_state)

        if self.q_target is not None:
            target_path = base_path / "q_target.pt"
            target_state = torch.load(target_path)
            self.q_target.load_state_dict(target_state)

        optimizer_path = base_path / "optimizer.pkl"
        optimizer_state_dict = torch.load(optimizer_path)["optimizer_state"]
        self.qlearning.optimizer.load_state_dict(optimizer_state_dict)

        replay_path = base_path / "replay.pkl"
        if replay_path.exists():
            self.replay_buffer = dill.loads(replay_path)
        if self.verbose:
            print(f"Loaded checkpoint at {path}")

    def adapt_hps(self, metrics):
        """Set hyperparameters."""
        metrics = super().adapt_hps(metrics)
        self.policy.epsilon = self._epsilon
        for g in self.qlearning.optimizer.param_groups:
            g["lr"] = self.learning_rate
