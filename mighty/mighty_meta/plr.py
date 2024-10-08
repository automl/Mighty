"""Curriculum Learning via Prioritized Level Replay.
This is adapted from: https://github.com/facebookresearch/level-replay/blob/main/level_replay/level_sampler.py.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class PrioritizedLevelReplay(MightyMetaComponent):
    """Curriculum Learning via Prioritized Level Replay."""

    def __init__(
        self,
        alpha=1.0,
        rho=0.2,
        staleness_coeff=0,
        sample_strategy="value_l1",
        score_transform="power",
        temperature=1.0,
        staleness_transform="power",
        staleness_temperature=1.0,
        eps=1e-3,
    ) -> None:
        """PLR initialization.

        :param alpha: Decay factor for scores
        :param rho: Minimum proportion of instances that has to be
            seen before re-sampling seen ones
        :param staleness_coeff: Staleness coefficient
        :param sample_strategy: Strategy for level sampling.
            One of: random, sequential, policy_entropy, least_confidence,
            min_margin, gae, value_l1, one_step_td_error
        :param score_transform: Transformation for the score.
            One of: max, constant, eps_greedy, rank, power softmax
        :param termperature: Temperature for score transformation
        :param staleness_transform: Transformation for staleness.
            One of: max, constant, eps_greedy, rank, power softmax
        :param staleness_temperature: Temperature for staleness transformation
        :return:
        """
        super().__init__()
        self.rng = np.random.default_rng()
        self.alpha = alpha
        self.rho = rho
        self.staleness_coef = staleness_coeff
        self.sample_strategy = sample_strategy
        self.eps = eps
        self.instance_scores = {}
        self.staleness = {}
        self.all_instances = None
        self.index = 0
        self.num_actions = None
        self.score_transform = score_transform
        self.temperature = temperature
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        self.pre_episode_methods = [self.get_instance]
        self.post_episode_methods = [self.add_rollout]

    def get_instance(self, metrics=None):
        """Get Training instance on episode start.

        :param metrics: Current metrics dict
        :return:
        """
        if self.sample_strategy == "random":
            instances = self.rng.choice(self.all_instances, size=self.num_instances)

        if self.sample_strategy == "sequential":
            instances = []
            for _ in range(self.num_instances):
                instances.append(self.all_instances[self.index])
                self.index = (self.index + 1) % len(self.all_instances)

        num_unseen = len(self.all_instances) - len(list(self.instance_scores.keys()))
        proportion_seen = (len(self.all_instances) - num_unseen) / len(
            self.all_instances
        )
        instances = []
        for _ in range(self.num_instances):
            if proportion_seen >= self.rho and self.rng.random() < proportion_seen:
                instances.append(self._sample_replay_level())
            else:
                instances.append(self._sample_unseen_level())
        metrics["env"].inst_ids = instances

    def _sample_replay_level(self):
        """Get already seen level.

        :return:
        """
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=float) / len(
                sample_weights
            )

        idx = self.rng.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[
            0
        ]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance

    def sample_weights(self):
        """Get weights for sampling.

        :return:
        """
        weights = self._score_transform(
            self.score_transform, self.temperature, self.instance_scores
        )
        ww = []
        for i, w in zip(self.all_instances, weights, strict=False):
            if i not in self.instance_scores:
                ww.append(0)
            else:
                ww.append(w)
        weights = np.array(ww)

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform, self.staleness_temperature, self.staleness
            )
            ws = []
            for i, w in zip(self.all_instances, staleness_weights, strict=False):
                if i not in self.instance_scores:
                    ws.append(0)
                else:
                    ws.append(w)
            staleness_weights = np.array(ws)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z

            weights = (
                1 - self.staleness_coef
            ) * weights + self.staleness_coef * staleness_weights

        return weights

    def _sample_unseen_level(self):
        """Get an unseen instance.

        :return:
        """
        sample_weights = np.zeros(len(self.all_instances))
        num_unseen = len(self.all_instances) - len(list(self.instance_scores.keys()))
        for c, i in enumerate(self.all_instances):
            if i not in self.instance_scores:
                sample_weights[c] = 1 / num_unseen
        idx = self.rng.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[
            0
        ]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance

    def _update_staleness(self, selected_id):
        """Update instance staleness.

        :param selected_id: instance id for which to update
        :return:
        """
        if self.staleness_coef > 0:
            self.staleness = {k: v + 1 for k, v in self.staleness.items()}
            self.staleness[selected_id] = 0

    def score_function(self, reward, values, logits):
        """Get score.

        :param reward: Rollout rewards
        :param values: Rollout values
        :param logits: Rollout logits
        :return: score
        """
        if self.sample_strategy == "random":
            score = 1
        elif self.sample_strategy == "policy_entropy":
            if logits is None:
                raise ValueError("Logits are required for policy entropy.")
            score = self._average_entropy(logits)
        elif self.sample_strategy == "least_confidence":
            if logits is None:
                raise ValueError("Logits are required for least confidence.")
            score = self._average_least_confidence(logits)
        elif self.sample_strategy == "min_margin":
            if logits is None:
                raise ValueError("Logits are required for min margin.")
            score = self._average_min_margin(logits)
        elif self.sample_strategy == "gae":
            score = self._average_gae(reward, values)
        elif self.sample_strategy == "value_l1":
            score = self._average_value_l1(reward, values)
        elif self.sample_strategy == "one_step_td_error":
            score = self._one_step_td_error(reward, values)
        else:
            raise NotImplementedError
        return score

    def add_rollout(self, metrics):
        """Save rollout stats.

        :param metrics: Current metrics dict
        :return:
        """
        instance_ids = metrics["env"].inst_ids
        episode_reward = metrics["episode_reward"]
        rollout_values = metrics["rollout_values"]
        rollout_logits = [None] * len(instance_ids)
        if "rollout_logits" in metrics:
            rollout_logits = metrics["rollout_logits"]

        if self.all_instances is None:
            self.all_instances = metrics["env"].instance_id_list
            self.num_instances = len(metrics["env"].inst_ids)
            for i in self.all_instances:
                if i not in self.instance_scores:
                    self.instance_scores[i] = 0
                if i not in self.staleness:
                    self.staleness[i] = 0
            if isinstance(metrics["env"].action_space, gym.spaces.Discrete):
                self.num_actions = metrics["env"].action_space.n

        for instance_id, ep_rew, rollouts, logits in zip(
            instance_ids, episode_reward, rollout_values, rollout_logits
        ):
            score = self.score_function(ep_rew, rollouts, logits)
            if instance_id not in self.instance_scores:
                self.instance_scores[instance_id] = 0
            old_score = self.instance_scores[instance_id]
            self.instance_scores[instance_id] = (
                1 - self.alpha
            ) * old_score + self.alpha * score

    def _average_entropy(self, episode_logits):
        """Get average entropy.

        :param episode_logits: Rollout logits
        :return: entropy
        """
        max_entropy = (
            -(1.0 / self.num_actions)
            * np.log(1.0 / self.num_actions)
            * self.num_actions
        )
        return (
            np.mean(np.sum(-np.exp(episode_logits) * episode_logits, axis=-1))
            / max_entropy
        )

    def _average_least_confidence(self, episode_logits):
        """Get least confidence.

        :param episode_logits: Rollout logits
        :return: least average confidence
        """
        return 1 - np.mean(np.exp(np.max(episode_logits, axis=-1)[0]))

    def _average_min_margin(self, episode_logits):
        """Get minimal margin.

        :param episode_logits: Rollout logits
        :return: min margin
        """
        top2_confidence = np.exp(episode_logits[np.argsort(episode_logits)[::-1]][:2])
        return 1 - np.mean(top2_confidence[:, 0] - top2_confidence[:, 1])

    def _average_gae(self, rewards, value_preds):
        """Get average gae.

        :param rewards: Rollout rewards
        :param value_preds: Rollout values
        :return: average_gae
        """
        advantages = rewards - value_preds
        return np.mean(advantages)

    def _average_value_l1(self, rewards, value_preds):
        """Get average value l1.

        :param rewards: Rollout rewards
        :param value_preds: Rollout values
        :return: average l1
        """
        advantages = rewards - value_preds
        return np.mean(abs(advantages))

    def _one_step_td_error(self, rewards, value_preds):
        """Get one step td error.

        :param rewards: Rollout rewards
        :param value_preds: Rollout values
        :return: td error
        """
        max_t = len(rewards)
        td_errors = rewards[:-1] + value_preds[: max_t - 1] - value_preds[1:max_t]
        return np.mean(abs(td_errors))

    def _score_transform(self, transform, temperature, scores):
        """Transform score.

        :param transform: Transformation to apply
        :param temperature: Transformation temperature
        :param scores: Scores to transform
        :return: scores
        """
        scores = np.array(list(scores.values()))
        if transform == "constant":
            weights = np.ones_like(scores)
        if transform == "max":
            weights = np.zeros_like(scores)
            scores = scores[:]
            for i in range(len(scores)):
                if i not in self.instance_scores:
                    scores[i] = -float("inf")
            argmax = self.rng.choice(np.flatnonzero(np.isclose(scores, max(scores))))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[np.argmax(scores)] = 1.0 - self.eps
            weights += self.eps / len(self.all_instances)
        elif transform == "rank":
            temp = np.flip(np.argsort(scores))
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1.0 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)

        return weights
