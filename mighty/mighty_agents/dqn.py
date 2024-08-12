"""DQN agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List

import dill  # type: ignore
import numpy as np
import torch
from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class
from mighty.mighty_exploration import EpsilonGreedy, MightyExplorationPolicy
from mighty.mighty_models import DQN
from mighty.mighty_update import DoubleQLearning, QLearning
from omegaconf import OmegaConf
from mighty.mighty_replay import TransitionBatch

if TYPE_CHECKING:
    from mighty.mighty_replay import MightyReplay
    from mighty.mighty_utils.logger import Logger
    from mighty.mighty_utils.types import TypeKwargs
    from omegaconf import DictConfig

    from mighty.mighty_utils.env_handling import MIGHTYENV


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
        env: MIGHTYENV,  # type: ignore
        logger: Logger,
        seed: int | None = None,
        eval_env: MIGHTYENV = None,  # type: ignore
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
        soft_update_weight: float = 0.01,
        policy_class: str | DictConfig | type[MightyExplorationPolicy] | None = None,
        policy_kwargs: TypeKwargs | None = None,
        q_class: str | DictConfig | type[DQN] | None = None,
        q_kwargs: TypeKwargs | None = None,
        td_update_class: type[QLearning] = QLearning,
        td_update_kwargs: TypeKwargs | None = None,
        save_replay: bool = False,
    ):
        # FIXME: the arguments are not complete. Double check all classes.
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
        q_class = retrieve_class(cls=q_class, default_cls=DQN)  # type: ignore
        if q_kwargs is None:
            q_kwargs = {"n_layers": 0}  # type: ignore
        self.q_class = q_class
        self.q_kwargs = q_kwargs

        # Policy Class
        policy_class = retrieve_class(cls=policy_class, default_cls=EpsilonGreedy)  # type: ignore
        if policy_kwargs is None:
            policy_kwargs = {"epsilon": 0.1}  # type: ignore
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

        self.td_update_class = retrieve_class(
            cls=td_update_class, default_cls=DoubleQLearning
        )
        if td_update_kwargs is None:
            td_update_kwargs = {"gamma": gamma}  # type: ignore
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
    def value_function(self) -> DQN:
        """Q-function."""
        return self.q  # type: ignore

    # FIXME: these were introduced to enable ES for parameters and only exist for DQN currently
    # If we want to keep the functionality, we should replicate the property in the other algorithms
    @property
    def parameters(self) -> List:
        """Q-function parameters."""
        if self.use_target:
            return list(self.q.parameters()) + list(self.q_target.parameters())  # type: ignore
        else:
            return list(self.q.parameters())  # type: ignore

    def _initialize_agent(self) -> None:
        """Initialize DQN specific things like q-function."""

        if not isinstance(self.q_kwargs, dict):
            self.q_kwargs = OmegaConf.to_container(self.q_kwargs)  # type: ignore

        self.q = self.q_class(  # type: ignore
            num_actions=self.env.single_action_space.n,  # type: ignore
            obs_size=self.env.single_observation_space.shape,  # type: ignore
            **self.q_kwargs,
        )
        self.policy = self.policy_class(algo="q", model=self.q, **self.policy_kwargs)  # type: ignore

        # target network
        if not self.use_target:
            self.q_target = None
        else:
            q_state = self.q.state_dict()
            self.q_target = self.q_class(
                num_actions=self.env.single_action_space.n,  # type: ignore
                obs_size=self.env.single_observation_space.shape,  # type: ignore
                **self.q_kwargs,
            )
            self.q_target.load_state_dict(q_state)

        # specify how to update value function
        self.qlearning = self.td_update_class(model=self.q, **self.td_update_kwargs)  # type: ignore
        # FIXME: I think we might want to replace all normal if statements:
        # 1. richt print in base agent + runners
        # 2. loggers everywhere else with configurable verbosity
        # Then we won't need to have verbose checks
        print("Initialized agent.")

    def update_agent(self, **kwargs) -> Any:  # type: ignore
        """Compute and apply TD update.

        :param step: Current training step
        :return:
        """

        transition_batch = self.buffer.sample(batch_size=self._batch_size)  # type: ignore
        preds, targets = self.qlearning.get_targets(  # type: ignore
            transition_batch, self.q, self.q_target
        )

        metrics_q = self.qlearning.apply_update(preds, targets)  # type: ignore
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
                self.q.parameters(),  # type: ignore
                self.q_target.parameters(),
                strict=False,
            ):
                target_param.data.copy_(
                    self.soft_update_weight * param.data
                    + (1 - self.soft_update_weight) * target_param.data
                )

        return metrics_q

    def process_transition(  # type: ignore
        self,
        curr_s,
        action,
        reward,
        next_s,
        dones,
        log_prob=None,
        metrics: Dict = None,  # type: ignore
    ) -> Dict:
        # convert into a transition object
        transition = TransitionBatch(curr_s, action, reward, next_s, dones)

        if "rollout_values" not in metrics:
            metrics["rollout_values"] = np.empty((0, self.env.single_action_space.n))  # type: ignore

        # Add Td-error to metrics
        metrics["td_error"] = (
            self.qlearning.td_error(transition, self.q, self.q_target).detach().numpy()  # type: ignore
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
        self.buffer.add(transition, metrics)  # type: ignore

        return metrics

    def save(self, t: int) -> None:
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
        torch.save(self.q.state_dict(), q_path)  # type: ignore

        # Save target parameters
        if self.q_target is not None:
            target_path = self.checkpoint_dir / "q_target.pt"
            torch.save(self.q_target.state_dict(), target_path)

        # Save optimizer state
        optimizer_path = self.checkpoint_dir / "optimizer.pkl"
        torch.save(
            {"optimizer_state": self.qlearning.optimizer.state_dict()},  # type: ignore
            optimizer_path,
        )

        # Save replay buffer
        if self.save_replay:
            replay_path = self.checkpoint_dir / "replay.pkl"
            self.buffer.save(replay_path)  # type: ignore

        if self.verbose:
            print(f"Saved checkpoint at {self.checkpoint_dir}")

    def load(self, path: str) -> None:
        """Set the internal state of the agent, e.g. after loading."""
        base_path = Path(path)
        q_path = base_path / "q.pt"
        q_state = torch.load(q_path)
        self.q.load_state_dict(q_state)  # type: ignore

        if self.q_target is not None:
            target_path = base_path / "q_target.pt"
            target_state = torch.load(target_path)
            self.q_target.load_state_dict(target_state)

        optimizer_path = base_path / "optimizer.pkl"
        optimizer_state_dict = torch.load(optimizer_path)["optimizer_state"]
        self.qlearning.optimizer.load_state_dict(optimizer_state_dict)  # type: ignore

        replay_path = base_path / "replay.pkl"
        if replay_path.exists():
            self.buffer = dill.loads(replay_path)
        if self.verbose:
            print(f"Loaded checkpoint at {path}")

    def adapt_hps(self, metrics: Dict) -> None:
        """Set hyperparameters."""
        super().adapt_hps(metrics)
        if "hp/soft_update_weight" in metrics:
            self.soft_update_weight = metrics["hp/soft_update_weight"]
        for g in self.qlearning.optimizer.param_groups:  # type: ignore
            g["lr"] = self.learning_rate

    # FIXME: what exactly do we use this for?
    # I know it was in the base agent ifs, but I think that's fundamentally not a good idea
    # I can see how something like on-policy vs off-policy would make sense though
    # Not sure whether we want to put this in the agent itself or in init
    # Pro agent: each class tells us what algo it is
    # Pro init: we can import a list of agents of a certain kind
    # Of course we could do it in the agent itself as a static attribute and check for it in init
    @property
    def agent_type(self) -> str:
        return "DQN"
