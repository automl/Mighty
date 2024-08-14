import torch
import torch.optim as optim
import torch.nn.functional as F
from mighty.mighty_models.sac import SACModel

from typing import Dict, Tuple
from mighty.mighty_replay.mighty_replay_buffer import TransitionBatch


# FIXME: we might want to move this to a general update utils module
def polyak_update(source_params, target_params, tau: float):  # type: ignore
    for target_param, param in zip(target_params, source_params):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class SACUpdate:
    def __init__(
        self,
        model: SACModel,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        value_lr: float = 0.001,
        tau: float = 0.005,
        alpha: float = 0.2,
        gamma: float = 0.99,
    ):
        """
        Initialize the SAC update mechanism.

        :param model: The SAC model containing policy and Q-networks.
        :param policy_lr: Learning rate for the policy network.
        :param q_lr: Learning rate for the Q-networks.
        :param value_lr: Learning rate for the value network.
        :param tau: Soft update parameter for the target networks.
        :param alpha: Entropy regularization coefficient.
        """
        self.model = model
        self.policy_optimizer = optim.Adam(model.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer1 = optim.Adam(model.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(model.q_net2.parameters(), lr=q_lr)
        self.value_optimizer = optim.Adam(model.value_net.parameters(), lr=value_lr)
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma

    def calculate_td_error(self, transition: TransitionBatch) -> Tuple:
        """Calculate the TD error for a given transition.

        :param transition: Current transition
        :return: TD error
        """
        with torch.no_grad():
            next_mean, next_log_std = self.model.forward_policy(
                torch.as_tensor(transition.next_obs, dtype=torch.float32)
            )

            next_std = next_log_std.exp()
            next_actions = torch.normal(
                next_mean, next_std
            )  # TODO: revisit action dimensionsa
            # FIXME: is this still open?

            next_log_probs = (
                -0.5
                * (
                    ((next_actions - next_mean) / (next_std + 1e-6)) ** 2
                    + 2 * next_log_std
                    + torch.log(torch.as_tensor(2.0) * torch.pi)
                )
            ).sum(dim=-1, keepdim=True)

            next_q1 = self.model.forward_q1(
                torch.cat(
                    [
                        torch.as_tensor(transition.next_obs, dtype=torch.float32),
                        next_actions,
                    ],
                    dim=-1,
                )
            )
            next_q2 = self.model.forward_q2(
                torch.cat(
                    [
                        torch.as_tensor(transition.next_obs, dtype=torch.float32),
                        next_actions,
                    ],
                    dim=-1,
                )
            )
            next_q = torch.min(next_q1, next_q2)

            target_q = transition.rewards.unsqueeze(-1) + (
                1 - transition.dones.unsqueeze(-1)
            ) * self.gamma * (next_q - self.alpha * next_log_probs)

        current_q1 = self.model.forward_q1(
            torch.cat(
                [
                    torch.as_tensor(transition.observations, dtype=torch.float32),
                    torch.as_tensor(transition.actions, dtype=torch.float32),
                ],
                dim=-1,
            )
        )
        current_q2 = self.model.forward_q2(
            torch.cat(
                [
                    torch.as_tensor(transition.observations, dtype=torch.float32),
                    torch.as_tensor(transition.actions, dtype=torch.float32),
                ],
                dim=-1,
            )
        )

        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q

        return td_error1, td_error2

    def update(self, batch: TransitionBatch) -> Dict:
        """
        Perform an update of the SAC model using a batch of experience.

        :param batch: A batch of experience data.
        :return: A dictionary of loss values for tracking.
        """
        states = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        # next_states = batch.next_obs
        dones = batch.dones

        # Compute target values for the Q-function using the target policy
        td_error1, td_error2 = self.calculate_td_error(batch)
        target_q = rewards.unsqueeze(-1) + (
            1 - dones.unsqueeze(-1)
        ) * self.gamma * torch.min(td_error1, td_error2)

        # Compute Q-function loss
        with torch.autograd.set_detect_anomaly(True):
            q1 = self.model.forward_q1(torch.cat([states, actions], dim=-1))
            q2 = self.model.forward_q2(torch.cat([states, actions], dim=-1))
            q_loss1 = F.mse_loss(q1, target_q)
            q_loss2 = F.mse_loss(q2, target_q)
            q_loss = q_loss1 + q_loss2

        # Compute policy loss
        new_mean, new_log_std = self.model.forward_policy(states)
        new_std = new_log_std.exp()
        new_actions = torch.normal(new_mean, new_std)
        log_probs = (
            -0.5
            * (
                ((new_actions - new_mean) / (new_std + 1e-6)) ** 2
                + 2 * new_log_std
                + torch.log(torch.tensor(2) * torch.pi)
            )
        ).sum(dim=-1, keepdim=True)

        q_new_actions = torch.min(
            self.model.forward_q1(torch.cat([states, new_actions], dim=-1)),
            self.model.forward_q2(torch.cat([states, new_actions], dim=-1)),
        )
        policy_loss = (self.alpha * log_probs - q_new_actions).mean()

        # Compute value loss
        value = self.model.forward_value(states)
        value_loss = F.mse_loss(value, target_q)

        # Combine all losses
        total_loss = q_loss + policy_loss + value_loss

        # Optimize all networks
        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.step()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        # Soft update Q-networks' target parameters
        if hasattr(self.model, "target_q_net1"):
            polyak_update(
                self.model.q_net1.parameters(),
                self.model.target_q_net1.parameters(),
                self.tau,
            )
        if hasattr(self.model, "target_q_net2"):
            polyak_update(
                self.model.q_net2.parameters(),
                self.model.target_q_net2.parameters(),
                self.tau,
            )

        return {
            "q_loss1": q_loss1.item(),
            "q_loss2": q_loss2.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
