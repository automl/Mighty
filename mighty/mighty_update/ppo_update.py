import torch
import torch.optim as optim
from typing import Dict
from mighty.mighty_models.ppo import PPOModel


class PPOUpdate:
    def __init__(
        self,
        model: PPOModel,
        policy_lr: float = 0.001,
        value_lr: float = 0.001,
        epsilon: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        """Initialize PPO update mechanism."""
        self.model = model
        self.policy_optimizer = optim.Adam(
            self.model.policy_net.parameters(), lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.model.value_net.parameters(), lr=value_lr
        )
        self.epsilon = epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the PPO model."""

        states = batch.observations.squeeze(0)
        actions = batch.actions.squeeze(0)
        old_log_probs = batch.log_probs.squeeze(0)
        advantages = batch.advantages.squeeze(0)
        returns = batch.returns.squeeze(0)

        # Compute the value loss
        values = self.model.forward_value(states).squeeze()
        value_loss = torch.nn.functional.mse_loss(returns, values)

        # Normalize advantage does not make sense if mini batchsize == 1, see GH issue #325
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the policy loss
        if self.model.continuous_action:
            means, stds = self.model(states)
            dist = torch.distributions.Normal(means, stds)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            logits = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

        ratios = torch.exp(log_probs - old_log_probs)
        if len(ratios.shape) > 2:
            ratios = ratios.squeeze(-1)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

        # Update the policy network
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.policy_net.parameters(), self.max_grad_norm
        )
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }
