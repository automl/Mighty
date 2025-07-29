from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
from omegaconf import DictConfig

from mighty.mighty_agents.base_agent import MightyAgent, retrieve_class, update_buffer
from mighty.mighty_exploration import MightyExplorationPolicy, StochasticPolicy
from mighty.mighty_exploration.mighty_exploration_policy import sample_nondeterministic_logprobs
from mighty.mighty_models.sac import SACModel
from mighty.mighty_replay import MightyReplay, TransitionBatch
from mighty.mighty_update import SACUpdate
from mighty.mighty_utils.mighty_types import MIGHTYENV, TypeKwargs


class MightySACAgent(MightyAgent):
    def __init__(
        self,
        output_dir: Path,
        env: MIGHTYENV,
        eval_env: Optional[MIGHTYENV] = None,
        seed: Optional[int] = None,
        # --- PPO-style network sizes ---
        n_policy_units: int = 64,
        # FIXME: not currently used, will be integrated
        n_critic_units: int = 64,
        soft_update_weight: float = 0.005,
        # --- Replay & update scheduling ---
        batch_size: int = 256,
        learning_starts: int = 10000,
        update_every: int = 50,
        n_gradient_steps: int = 1,
        # --- Learning rates ---
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        # --- SAC hyperparameters ---
        gamma: float = 0.99,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        alpha_lr: float = 3e-4,
        # --- Network architecture (optional override) ---
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        log_std_min: float = -20,
        log_std_max: float = 2,
        # --- Logging & buffer ---
        render_progress: bool = True,
        log_wandb: bool = False,
        wandb_kwargs: Optional[Dict] = None,
        replay_buffer_class: Type[MightyReplay] = MightyReplay,
        replay_buffer_kwargs: Optional[TypeKwargs] = None,
        meta_methods: Optional[List[Union[str, type]]] = None,
        meta_kwargs: Optional[List[TypeKwargs]] = None,
        policy_class: Optional[
            Union[str, DictConfig, Type[MightyExplorationPolicy]]
        ] = None,
        policy_kwargs: Optional[Dict] = None,
        normalize_obs: bool = False,  # ← NEW
        normalize_reward: bool = False,  # ← NEW (optional)
    ):
        """Initialize SAC agent with tunable hyperparameters and backward-compatible names."""
        # Map PPO-style units to hidden_sizes if not provided
        # FIXME: This should be replaced by a more flexible architecture definition like in DQN
        # See https://github.com/automl/Mighty/issues/57
        if hidden_sizes is None:
            hidden_sizes = [n_policy_units, n_policy_units]
        tau = soft_update_weight

        # Save hyperparameters
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_every = update_every
        self.n_gradient_steps = n_gradient_steps
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr

        # Placeholders for model and updater
        self.model: SACModel | None = None
        self.update_fn: SACUpdate | None = None

        # Exploration policy class
        self.policy_class = retrieve_class(
            cls=policy_class, default_cls=StochasticPolicy
        )
        self.policy_kwargs = policy_kwargs or {
            "discrete": False  # Default to continuous SAC
        }

        super().__init__(
            env=env,
            output_dir=output_dir,
            seed=seed,
            eval_env=eval_env,
            learning_starts=learning_starts,
            n_gradient_steps=n_gradient_steps,
            render_progress=render_progress,
            log_wandb=log_wandb,
            wandb_kwargs=wandb_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            meta_methods=meta_methods,
            meta_kwargs=meta_kwargs,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
        )

        # Initialize loss buffer for logging
        self.loss_buffer = {
            "q_loss1": [],
            "q_loss2": [],
            "policy_loss": [],
            "td_error1": [],
            "td_error2": [],
            "step": [],
        }

    def _initialize_agent(self) -> None:
        # Determine dimensions
        obs_dim = self.env.single_observation_space.shape[0]
        act_dim = self.env.single_action_space.shape[0]

        # Build the SACModel
        self.model = SACModel(
            obs_size=obs_dim,
            action_size=act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )

        # Exploration policy wrapper
        self.policy = self.policy_class(
            algo=self, model=self.model, **self.policy_kwargs
        )

        # Updater
        self.update_fn = SACUpdate(
            model=self.model,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            tau=self.tau,
            alpha=self.alpha,
            gamma=self.gamma,
            auto_alpha=self.auto_alpha,
            target_entropy=self.target_entropy,
            alpha_lr=self.alpha_lr,
        )

    @property
    def parameters(self) -> List[torch.nn.Parameter]:
        """Collect policy + Q‐network parameters for SAC."""
        return (
            list(self.model.policy_net.parameters())
            + list(self.model.q_net1.parameters())
            + list(self.model.q_net2.parameters())
        )

    def update_agent(self, *args, **kwargs) -> Dict[str, float]:
        # Only update at intervals after warmup
        if self.steps < self.learning_starts or self.steps % self.update_every != 0:
            return {}

        # Accumulate metrics over multiple gradient steps
        metrics_acc: Dict[str, float] = {}
        for _ in range(self.n_gradient_steps):
            batch = self.buffer.sample(self.batch_size)
            metrics = self.update_fn.update(batch)
            for k, v in metrics.items():
                metrics_acc.setdefault(k, 0.0)
                metrics_acc[k] += v
        # Average
        for k in metrics_acc:
            metrics_acc[k] /= self.n_gradient_steps

        # Log to buffer
        stats = {**metrics_acc, "step": self.steps}
        self.loss_buffer = update_buffer(self.loss_buffer, stats)

        return metrics_acc

    def process_transition(
        self,
        curr_s,
        action,
        reward,
        next_s,
        dones,
        log_prob=None,
        metrics: Optional[Dict] = None,
    ) -> Dict:
        # Ensure metrics dict
        if metrics is None:
            metrics = {}
        # Pack transition
        transition = TransitionBatch(curr_s, action, reward, next_s, dones)
        # Compute per-transition TD errors for logging
        td1, td2 = self.update_fn.calculate_td_error(transition)
        metrics["td_error1"] = td1.detach().cpu().numpy()
        metrics["td_error2"] = td2.detach().cpu().numpy()
        # Add to replay buffer
        self.buffer.add(transition, metrics)
        return metrics

    @property
    def value_function(self) -> torch.nn.Module:
        """Value function for compatibility: V(s) = min(Q1,Q2)(s, a_policy) - alpha * log_pi(a|s)."""

        # Lazily create a wrapper module
        class _ValueFunction(torch.nn.Module):
            def __init__(self, agent):
                super().__init__()
                self.agent = agent

            def forward(self, state):
                state_t = torch.as_tensor(state, dtype=torch.float32)
                with torch.no_grad():
                    # deterministic policy action
                    a, z, mean, log_std = self.agent.model(state_t, deterministic=True)
                    logp = sample_nondeterministic_logprobs(z, mean, log_std, sac=True)
                    sa = torch.cat([state_t, a], dim=-1)
                    q1 = self.agent.model.q_net1(sa)
                    q2 = self.agent.model.q_net2(sa)
                    return torch.min(q1, q2) - self.agent.alpha * logp

        if not hasattr(self, "_vf_module"):
            self._vf_module = _ValueFunction(self)
        return self._vf_module

    def save(self, t: int) -> None:
        super().make_checkpoint_dir(t)
        torch.save(
            self.model.policy_net.state_dict(), self.checkpoint_dir / "policy_net.pt"
        )
        torch.save(self.model.q_net1.state_dict(), self.checkpoint_dir / "q_net1.pt")
        torch.save(self.model.q_net2.state_dict(), self.checkpoint_dir / "q_net2.pt")

    def load(self, path: str) -> None:
        base = Path(path)
        self.model.policy_net.load_state_dict(torch.load(base / "policy_net.pt"))
        self.model.q_net1.load_state_dict(torch.load(base / "q_net1.pt"))
        self.model.q_net2.load_state_dict(torch.load(base / "q_net2.pt"))
