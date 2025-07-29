from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from mighty.mighty_models.sac import SACModel
from mighty.mighty_replay.mighty_replay_buffer import TransitionBatch
from mighty.mighty_utils.update_utils import polyak_update


class SACUpdate:
    def __init__(
        self,
        model: SACModel,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        tau: float = 0.005,
        alpha: float = 0.2,
        gamma: float = 0.99,
        target_entropy: float = None,
        auto_alpha: bool = True,
        alpha_lr: float = 3e-4,
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

        # optimizers
        self.policy_optimizer = optim.Adam(model.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer1 = optim.Adam(model.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(model.q_net2.parameters(), lr=q_lr)
        # hyperparams
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.action_dim = self.model.action_size

        if self.auto_alpha:
            # log_alpha is a torch Parameter so it will be optimized
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            # Default: -action_dim, but can override
            if target_entropy is None:
                self.target_entropy = -float(self.action_dim)
            else:
                self.target_entropy = float(target_entropy)
        else:
            self.alpha = alpha

    def calculate_td_error(self, transition: TransitionBatch) -> Tuple:
        """Calculate the TD error for a given transition.

        :param transition: Current transition
        :return: TD error
        """
        with torch.no_grad():
            # sample next action and log-prob
            _, z_next, mean_next, log_std_next = self.model(
                torch.as_tensor(transition.next_obs, dtype=torch.float32)
            )
            logp_next = self.model.policy_log_prob(z_next, mean_next, log_std_next)
            sa_next = torch.cat(
                [
                    torch.as_tensor(transition.next_obs, dtype=torch.float32),
                    torch.tanh(z_next),
                ],
                dim=-1,
            )
            # target Q from target networks
            q1_t = self.model.target_q_net1(sa_next)
            q2_t = self.model.target_q_net2(sa_next)
            alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
            q_target = torch.as_tensor(
                transition.rewards, dtype=torch.float32
            ).unsqueeze(-1) + (
                1 - torch.as_tensor(transition.dones, dtype=torch.float32).unsqueeze(-1)
            ) * self.gamma * (torch.min(q1_t, q2_t) - alpha * logp_next)
        # current Q estimates
        sa = torch.cat(
            [
                torch.as_tensor(transition.observations, dtype=torch.float32),
                torch.as_tensor(transition.actions, dtype=torch.float32),
            ],
            dim=-1,
        )
        q1_curr = self.model.q_net1(sa)
        q2_curr = self.model.q_net2(sa)
        td_error1 = q1_curr - q_target
        td_error2 = q2_curr - q_target
        return td_error1, td_error2

    def update(self, batch: TransitionBatch) -> Dict:
        """
        Perform an update of the SAC model using a batch of experience.

        :param batch: A batch of experience data.
        :return: A dictionary of loss values for tracking.
        """

        states = torch.as_tensor(batch.observations, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).unsqueeze(-1)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.as_tensor(batch.next_obs, dtype=torch.float32)

        # --- Q-network update ---
        with torch.no_grad():
            # BUG: this uses `states` but should use `next_states`
            a_next, z_next, mean_next, log_std_next = self.model(next_states)
            logp_next = self.model.policy_log_prob(z_next, mean_next, log_std_next)
            sa_next = torch.cat([next_states, a_next], dim=-1)
            q1_t = self.model.target_q_net1(sa_next)
            q2_t = self.model.target_q_net2(sa_next)
            q_target = rewards + (1 - dones) * self.gamma * (
                torch.min(q1_t, q2_t) - self.alpha * logp_next
            )

        # current estimates and losses
        sa = torch.cat([states, actions], dim=-1)
        q1 = self.model.q_net1(sa)
        q2 = self.model.q_net2(sa)
        q_loss1 = F.mse_loss(q1, q_target)
        q_loss2 = F.mse_loss(q2, q_target)
        q_loss = q_loss1 + q_loss2

        # --- Policy update ---
        a, z, mean, log_std = self.model(states)
        logp = self.model.policy_log_prob(z, mean, log_std)
        sa_pi = torch.cat([states, a], dim=-1)
        q1_pi = self.model.q_net1(sa_pi)
        q2_pi = self.model.q_net2(sa_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        if self.auto_alpha:
            self.alpha = self.log_alpha.exp()

        policy_loss = (self.alpha * logp - q_pi).mean()

        loss = q_loss + policy_loss

        # Zero ALL optimizers up-front
        for opt in (self.q_optimizer1, self.q_optimizer2, self.policy_optimizer):
            opt.zero_grad()

        loss.backward()

        for opt in (self.q_optimizer1, self.q_optimizer2, self.policy_optimizer):
            opt.step()

        # --- Entropy coefficient (alpha) update ---
        if self.auto_alpha:
            # Update alpha to adjust entropy toward target
            alpha_loss = -(
                self.log_alpha * (logp.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # alpha_value = self.log_alpha.exp().item()
        else:
            alpha_loss = 0.0
            # alpha_value = self.alpha

        # --- Soft update targets ---
        polyak_update(
            self.model.q_net1.parameters(),
            self.model.target_q_net1.parameters(),
            self.tau,
        )
        polyak_update(
            self.model.q_net2.parameters(),
            self.model.target_q_net2.parameters(),
            self.tau,
        )

        # --- Logging metrics ---
        td1, td2 = self.calculate_td_error(batch)
        return {
            "q_loss1": q_loss1.item(),
            "q_loss2": q_loss2.item(),
            "policy_loss": policy_loss.item(),
            # average td error per batch
            "td_error1": td1.mean().item(),
            "td_error2": td2.mean().item(),
        }
