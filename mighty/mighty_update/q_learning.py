"""Q-learning update (modified to accept nested optimizer_kwargs and max_grad_norm)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig


class QLearning:
    """Q-learning update."""

    def __init__(
        self,
        model,
        gamma: float,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        """Initialize the Q-learning update.

        :param model: The Q-network to optimize.
        :param gamma: Discount factor.
        :param optimizer_class: Optimizer class (e.g. torch.optim.Adam).
        :param optimizer_kwargs: Keyword args to pass into optimizer.
        :param max_grad_norm: If provided, gradient norms will be clipped to this value.
        """
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        # self.optimizer = optimizer_class(params=model.parameters(), **optimizer_kwargs)

        if isinstance(optimizer_class, DictConfig):
            # Hydra DictConfig typically has {"_target_": "torch.optim.Adam", "lr": 3e-4, ...}
            # instantiate() will call that class with any fields inside, so we pass params=model.parameters() too.
            self.optimizer = instantiate(optimizer_class, params=model.parameters())
        elif isinstance(optimizer_class, str):
            # If it's a string "torch.optim.Adam" (no nested kwargs), get the class then call it:
            cls = get_class(optimizer_class)
            self.optimizer = cls(params=model.parameters(), **optimizer_kwargs)
        else:
            # Already a Python class (e.g. torch.optim.Adam)
            self.optimizer = optimizer_class(
                params=model.parameters(), **optimizer_kwargs
            )

    def apply_update(self, preds, targets):
        """Apply the Q-learning update."""
        self.optimizer.zero_grad()
        loss = F.mse_loss(preds, targets)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], self.max_grad_norm
            )
        self.optimizer.step()
        return {"Update/loss": loss.detach().numpy().item()}

    def get_targets(self, batch, q_net, target_net=None):
        """Get targets for the Q-learning update."""
        if target_net is None:
            target_net = q_net
        max_next = (
            target_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .max(1)[0]
            .unsqueeze(1)
        )

        nonterminal_mask = 1.0 - batch.dones.unsqueeze(-1).to(torch.float32)
        targets = batch.rewards.unsqueeze(-1) + nonterminal_mask * self.gamma * max_next

        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)

    def td_error(self, batch, q_net, target_net=None):
        """Compute the TD error for the Q-learning update."""
        preds, targets = self.get_targets(batch, q_net, target_net)
        # td_errors = (targets - preds).squeeze(-1).detach()
        td_errors = F.mse_loss(preds, targets, reduction="none").detach().mean(axis=1)
        return td_errors


class DoubleQLearning(QLearning):
    """Double Q-learning update."""

    def __init__(
        self,
        model,
        gamma: float,
        optimizer_class=torch.optim.Adam,  # inherits new signature
        optimizer_kwargs: dict | None = None,  # inherits new signature
        max_grad_norm: float | None = None,  # inherits new signature
    ) -> None:
        """Initialize the Double Q-learning update."""
        super().__init__(model, gamma, optimizer_class, optimizer_kwargs, max_grad_norm)

    def get_targets(self, batch, q_net, target_net=None):
        if target_net is None:
            target_net = q_net
        argmax_a = (
            q_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .argmax(dim=1)
            .unsqueeze(-1)
        )
        max_next = target_net(
            torch.as_tensor(batch.next_obs, dtype=torch.float32)
        ).gather(1, argmax_a)

        nonterminal_mask = 1.0 - batch.dones.unsqueeze(-1).to(torch.float32)
        targets = batch.rewards.unsqueeze(-1) + nonterminal_mask * self.gamma * max_next
        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)


class ClippedDoubleQLearning(QLearning):
    """Clipped Double Q-learning update."""

    def __init__(
        self,
        model,
        gamma: float,
        optimizer_class=torch.optim.Adam,  # inherits new signature
        optimizer_kwargs: dict | None = None,  # inherits new signature
        max_grad_norm: float | None = None,  # inherits new signature
    ) -> None:
        """Initialize the Clipped Double Q-learning update."""
        super().__init__(model, gamma, optimizer_class, optimizer_kwargs, max_grad_norm)

    def get_targets(self, batch, q_net, target_net=None):
        if target_net is None:
            target_net = q_net
        argmax_a = (
            q_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .argmax(dim=1)
            .unsqueeze(-1)
        )
        max_next = q_net(torch.as_tensor(batch.next_obs, dtype=torch.float32)).gather(
            1, argmax_a
        )
        max_next_target = target_net(
            torch.as_tensor(batch.next_obs, dtype=torch.float32)
        ).gather(1, argmax_a)
        targets = batch.rewards.unsqueeze(-1) + (
            ~batch.dones.unsqueeze(-1)
        ) * self.gamma * torch.minimum(max_next_target, max_next)
        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)
