from typing import Dict

import torch
import torch.optim as optim

from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_replay.mighty_rollout_buffer import MaxiBatch


class PPOUpdate:
    def __init__(
        self,
        model: PPOModel,
        policy_lr: float = 3e-4,
        value_lr: float = 3e-4,
        epsilon: float = 0.1,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        minibatch_size: int = 32,
        kl_target: float = 0.01,
        use_value_clip: bool = True,
        value_clip_eps: float = 0.2,
        total_timesteps: int = 1_000_000,
        adaptive_lr: bool = True,
        min_lr: float = 1e-6,
    ):
        """Initialize PPO update mechanism."""
        self.model = model
        self.epsilon = epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.kl_target = kl_target
        self.use_value_clip = use_value_clip
        self.value_clip_eps = value_clip_eps
        self.adaptive_lr = adaptive_lr
        self.min_lr = min_lr

        self.total_steps = total_timesteps

        # Store initial learning rates
        self.initial_policy_lr = policy_lr
        self.initial_value_lr = value_lr

        # Optimizers
        policy_params = list(model.policy_head.parameters())
        value_params = list(model.value_head.parameters())

        extra_params = []
        if getattr(model, "continuous_action", False) and hasattr(model, "log_std"):
            extra_params.append(model.log_std)

        self.optimizer = optim.Adam(
            [
                {"params": policy_params, "lr": policy_lr},
                {"params": value_params, "lr": value_lr},
                *([{"params": extra_params, "lr": policy_lr}] if extra_params else []),
            ],
            eps=1e-5,
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(
                0.1, 1 - step / float(self.total_steps)
            ),  # Don't go below 10% of initial
        )

    def update(self, batch: MaxiBatch) -> Dict[str, float]:
        """
        PPO update continuous branch now evaluates π_new on the *old* latent
        actions, which is required for a correct importance-sampling ratio.
        Assumes each minibatch has a `latents` Tensor containing the pre-tanh
        actions (`z_old`).  If you do not store that in your buffer yet, either:
            • add it during rollout  (recommended), or
            • reconstruct it on-the-fly via atanh(squashed_action.clamp(-0.999,0.999)).
        """

        # ─────────────────── cache old values & log-probs ───────────────────
        with torch.no_grad():
            # squeeze(-1): value head outputs (..., 1); returns/advantages are (..., ),
            # so without it (returns - values) broadcasts to a (B, B) matrix.
            old_values = [
                self.model.forward_value(mb.observations).squeeze(-1)
                for mb in batch.minibatches
            ]
            old_log_probs = [mb.log_probs.clone() for mb in batch.minibatches]

        # ───────────────────── advantage normalisation ──────────────────────
        flat_adv = batch.advantages.view(-1)
        adv_mean, adv_std = flat_adv.mean(), flat_adv.std() + 1e-8

        metrics, mb_updates = (
            {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            },
            0,
        )

        # ───────────────────────── main PPO loop ────────────────────────────
        for epoch in range(self.n_epochs):
            epoch_kls = []
            for i, mb in enumerate(batch.minibatches):
                adv = ((mb.advantages - adv_mean) / adv_std).detach()

                # ---- value loss ---------------------------------------------------
                values = self.model.forward_value(mb.observations).squeeze(-1)
                if self.use_value_clip:
                    clipped = old_values[i] + (values - old_values[i]).clamp(
                        -self.value_clip_eps, self.value_clip_eps
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (mb.returns - values).pow(2), (mb.returns - clipped).pow(2)
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * (mb.returns - values).pow(2).mean()

                # ---- policy loss  (continuous & discrete share the same surr) ----
                if self.model.continuous_action:
                    # Get model output
                    model_output = self.model(mb.observations)

                    # NEW: Handle both modes
                    if hasattr(self.model, "tanh_squash") and self.model.tanh_squash:
                        # Tanh squashing mode (existing logic)
                        _, _, mean, log_std = model_output  # 4-tuple
                        dist = torch.distributions.Normal(mean, log_std.exp())

                        z_old = mb.latents  # stored pre-tanh
                        log_pz = dist.log_prob(z_old).sum(-1)
                        log_corr = torch.log(1 - torch.tanh(z_old).pow(2) + 1e-6).sum(
                            -1
                        )
                        log_probs = log_pz - log_corr

                    else:
                        # Standard PPO mode (new logic)
                        _, mean, log_std = model_output  # 3-tuple
                        dist = torch.distributions.Normal(mean, log_std.exp())

                        # Direct log prob on actions (no latents needed)
                        log_probs = dist.log_prob(mb.actions).sum(-1)

                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits = self.model(mb.observations)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(mb.actions)
                    entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - old_log_probs[i])
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
                p_loss = -torch.min(surr1, surr2).mean()

                # ---- combined loss & optimisation --------------------------------
                loss = p_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # ---- KL divergence on *same* z_old --------------------------------
                with torch.no_grad():
                    if self.model.continuous_action:
                        model_output_new = self.model(mb.observations)

                        if (
                            hasattr(self.model, "tanh_squash")
                            and self.model.tanh_squash
                        ):
                            # Tanh squashing mode
                            _, _, mean_new, log_std_new = model_output_new
                            dist_new = torch.distributions.Normal(
                                mean_new, log_std_new.exp()
                            )
                            log_pz_new = dist_new.log_prob(z_old).sum(-1)
                            log_corr_n = torch.log(
                                1 - torch.tanh(z_old).pow(2) + 1e-6
                            ).sum(-1)
                            new_lp = log_pz_new - log_corr_n
                        else:
                            # Standard PPO mode
                            _, mean_new, log_std_new = model_output_new
                            dist_new = torch.distributions.Normal(
                                mean_new, log_std_new.exp()
                            )
                            new_lp = dist_new.log_prob(mb.actions).sum(-1)
                    else:
                        logits_new = self.model(mb.observations)
                        new_lp = torch.distributions.Categorical(
                            logits=logits_new
                        ).log_prob(mb.actions)

                kl = (old_log_probs[i] - new_lp).mean()
                epoch_kls.append(kl)

                # ---- bookkeeping --------------------------------------------------
                metrics["policy_loss"] += p_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["entropy"] += entropy.item()
                mb_updates += 1

            # ───────── epoch end: LR adaptation, early stop, logging ─────────
            if len(epoch_kls) > 0:
                mean_kl = torch.stack(epoch_kls).mean()
            else:
                # If no minibatches were processed, set a default KL value
                mean_kl = torch.tensor(0.0)
                print("Warning: No minibatches processed in this epoch")

            # adaptive LR
            if self.adaptive_lr and self.kl_target:
                for g in self.optimizer.param_groups[:2]:  # policy & value groups
                    if mean_kl > 1.5 * self.kl_target:
                        g["lr"] = max(g["lr"] * 0.8, self.min_lr)
                    elif mean_kl < 0.5 * self.kl_target and epoch == 0:
                        g["lr"] = min(
                            g["lr"] * 1.1,
                            (
                                self.initial_policy_lr
                                if g is self.optimizer.param_groups[0]
                                else self.initial_value_lr
                            ),
                        )

                # early-stop if KL already large
                if mean_kl > self.kl_target:
                    break

        # Scheduler AFTER adaptive block (won’t drive LR below 0.1× init)
        self.scheduler.step()

        # clip param-group LR to min_lr
        for g in self.optimizer.param_groups:
            if g["lr"] < self.min_lr:
                g["lr"] = self.min_lr

        # final averaged metrics
        for k in metrics:
            metrics[k] /= mb_updates if mb_updates > 0 else 1

        metrics["approx_kl"] = mean_kl.item()  # final KL of the run

        return {
            "Update/policy_loss": metrics["policy_loss"],
            "Update/value_loss": metrics["value_loss"],
            "Update/entropy": metrics["entropy"],
            "Update/approx_kl": metrics["approx_kl"],
        }
