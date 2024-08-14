"""Q-learning update."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class QLearning:
    """Q-learning update."""

    def __init__(
        self, model, gamma, optimizer=torch.optim.Adam, **optimizer_kwargs
    ) -> None:
        """Initialize the Q-learning update."""
        self.gamma = gamma
        self.optimizer = optimizer(params=model.parameters(), **optimizer_kwargs)

    def apply_update(self, preds, targets):
        """Apply the Q-learning update."""
        self.optimizer.zero_grad()
        loss = F.mse_loss(preds, targets)
        loss.backward()
        self.optimizer.step()
        return {"Q-Update/loss": loss.detach().numpy().item()}

    def get_targets(self, batch, q_net, target_net=None):
        """Get targets for the Q-learning update."""
        if target_net is None:
            target_net = q_net
        max_next = (
            target_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .max(1)[0]
            .unsqueeze(1)
        )
        targets = (
            batch.rewards.unsqueeze(-1)
            + (~batch.dones.unsqueeze(-1)) * self.gamma * max_next
        )
        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)

    def td_error(self, batch, q_net, target_net=None):
        """Compute the TD error for the Q-learning update."""
        preds, targets = self.get_targets(batch, q_net, target_net)
        return F.mse_loss(preds, targets, reduction="none").detach().mean(axis=1)


class DoubleQLearning(QLearning):
    """Double Q-learning update."""

    def __init__(
        self, model, gamma, optimizer=torch.optim.Adam, **optimizer_kwargs
    ) -> None:
        """Initialize the Double Q-learning update."""
        super().__init__(model, gamma, optimizer, **optimizer_kwargs)

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
        targets = (
            batch.rewards.unsqueeze(-1)
            + (~batch.dones.unsqueeze(-1)) * self.gamma * max_next
        )
        preds = q_net(torch.as_tensor(batch.observations, dtype=torch.float32)).gather(
            1, batch.actions.to(torch.int64).unsqueeze(-1)
        )
        return preds.to(torch.float32), targets.to(torch.float32)


class ClippedDoubleQLearning(QLearning):
    """Clipped Double Q-learning update."""

    def __init__(
        self, model, gamma, optimizer=torch.optim.Adam, **optimizer_kwargs
    ) -> None:
        """Initialize the Clipped Double Q-learning update."""
        super().__init__(model, gamma, optimizer, **optimizer_kwargs)

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


class SPRQLearning(QLearning):
    """SPR Q-learning update."""

    def __init__(
        self,
        model,
        gamma,
        optimizer=torch.optim.Adam,
        spr_loss_weight=1,
        huber_delta=1,
        **optimizer_kwargs,
    ) -> None:
        """Initialize the SPR Q-learning update."""
        super().__init__(model, gamma, optimizer, **optimizer_kwargs)
        self.spr_loss_weight = spr_loss_weight
        self.huber_delta = huber_delta

    def get_targets(self, batch, buffer, q_net, target_net=None):
        if target_net is None:
            target_net = q_net
        max_next = (
            target_net(torch.as_tensor(batch.next_obs, dtype=torch.float32))
            .max(1)[0]
            .unsqueeze(1)
        )
        td_targets = batch.rewards + (~batch.dones) * self.gamma * max_next
        q_preds, spr_preds = q_net(
            torch.as_tensor(batch.observations, dtype=torch.float32),
            actions=batch.actions,
        )
        q_preds = q_preds.gather(1, batch.actions)
        spr_targets = target_net.project(buffer.next_states)
        spr_preds = spr_preds / np.linalg_norm(spr_preds, 2, -1, keepdims=True)
        spr_targets = spr_targets.reshape(-1)
        spr_targets = spr_targets / np.linalg_norm(spr_targets, 2, -1, keepdims=True)
        return (q_preds.to(torch.float32), spr_preds.to(torch.float32)), (
            td_targets.to(torch.float32),
            spr_targets.to(torch.float32),
        )

    def apply_update(self, preds, targets, optimizer):
        q_preds, spr_preds = preds
        td_targets, spr_targets = targets
        spr_loss = 0.5 * (spr_targets - spr_preds) ** 2
        dqn_loss = np.abs(td_targets - q_preds)
        dqn_loss = np.where(
            dqn_loss <= self.huber_delta,
            0.5 * dqn_loss**2,
            0.5 * self.huber_delta**2
            + self.huber_delta * (dqn_loss - self.huber_delta),
        )
        loss = dqn_loss + self.spr_loss_weight * spr_loss
        self.optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "Q-Update/total_loss": loss.detach(),
            "Q-Update/dqn_loss": dqn_loss.detach(),
            "Q-Update/SPR_loss": spr_loss.detach(),
        }
