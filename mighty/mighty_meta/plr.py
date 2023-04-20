# This is adapted from: https://github.com/facebookresearch/level-replay/blob/main/level_replay/level_sampler.py
import numpy as np
import gymnasium as gym
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class PrioritizedLevelReplay(MightyMetaComponent):
    def __init__(
        self,
        alpha=1.0,
        rho=0.2,
        nu=0.5,
        staleness_coeff=0,
        sample_strategy="value_l1",
        replay_schedule="proportional",
        score_transform="power",
        temperature=1.0,
        staleness_transform="power",
        staleness_temperature=1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.rho = rho
        self.nu = nu
        self.staleness_coef = staleness_coeff
        self.sample_strategy = sample_strategy
        self.replay_schedule = replay_schedule
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
        if self.all_instances is None:
            self.all_instances = metrics["env"].instance_id_list
            for i in self.all_instances:
                if i not in self.instance_scores.keys():
                    self.instance_scores[i] = 0
                if i not in self.staleness.keys():
                    self.staleness[i] = 0
            if isinstance(metrics["env"].action_space, gym.spaces.Discrete):
                self.num_actions = metrics["env"].action_space.n

        if self.sample_strategy == "random":
            instance = np.random.choice(self.all_instances)
            return instance

        if self.sample_strategy == "sequential":
            instance = self.all_instances[self.index]
            self.index = (self.index + 1) % len(self.all_instances)
            return instance

        num_unseen = len(self.all_instances) - len(list(self.instance_scores.keys()))
        proportion_seen = (len(self.all_instances) - num_unseen) / len(
            self.all_instances
        )

        if self.replay_schedule == "fixed":
            if proportion_seen >= self.rho:
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else:  # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=float) / len(
                sample_weights
            )

        idx = np.random.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[
            0
        ]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance

    def sample_weights(self):
        weights = self._score_transform(
            self.score_transform, self.temperature, self.instance_scores
        )
        ww = []
        for i, w in zip(self.all_instances, weights):
            if i not in self.instance_scores.keys():
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
            for i, w in zip(self.all_instances, staleness_weights):
                if i not in self.instance_scores.keys():
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
        sample_weights = np.zeros(len(self.all_instances))
        num_unseen = len(self.all_instances) - len(list(self.instance_scores.keys()))
        for c, i in enumerate(self.all_instances):
            if i not in self.instance_scores.keys():
                sample_weights[c] = 1 / num_unseen
        idx = np.random.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[
            0
        ]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance

    def _update_staleness(self, selected_id):
        if self.staleness_coef > 0:
            self.staleness = {k: v + 1 for k, v in self.staleness.items()}
            self.staleness[selected_id] = 0

    def score_function(self, reward, values, logits):
        if self.sample_strategy == "random":
            return 1
        elif self.sample_strategy == "policy_entropy":
            return self._average_entropy(logits)
        elif self.sample_strategy == "least_confidence":
            return self._average_least_confidence(logits)
        elif self.sample_strategy == "min_margin":
            return self._average_min_margin(logits)
        elif self.sample_strategy == "gae":
            return self._average_gae(reward, values)
        elif self.sample_strategy == "value_l1":
            return self._average_value_l1(reward, values)
        elif self.sample_strategy == "one_step_td_error":
            return self._one_step_td_error(reward, values)
        else:
            raise NotImplementedError

    def add_rollout(self, metrics):
        instance_id = metrics["env"].inst_id
        episode_reward = metrics["episode_reward"]
        rollout_values = metrics["rollout_values"]
        rollout_logits = None
        if "rollout_logits" in metrics.keys():
            rollout_logits = metrics["rollout_logits"]

        score = self.score_function(episode_reward, rollout_values, rollout_logits)
        old_score = self.instance_scores[instance_id]
        self.instance_scores[instance_id] = (
            1 - self.alpha
        ) * old_score + self.alpha * score

    def _average_entropy(self, episode_logits):
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
        return 1 - np.mean(np.exp(np.max(episode_logits, axis=-1)[0]))

    def _average_min_margin(self, episode_logits):
        top2_confidence = np.exp(episode_logits[np.argsort(episode_logits)[::-1]][:2])
        return 1 - np.mean((top2_confidence[:, 0] - top2_confidence[:, 1]))

    def _average_gae(self, rewards, value_preds):
        advantages = rewards - value_preds
        return np.mean(advantages)

    def _average_value_l1(self, rewards, value_preds):
        advantages = rewards - value_preds
        return np.mean(abs(advantages))

    def _one_step_td_error(self, rewards, value_preds):
        max_t = len(rewards)
        td_errors = (
            rewards[:-1] + value_preds[: max_t - 1] - value_preds[1:max_t]
        ).abs()
        return np.mean(abs(td_errors))

    def _score_transform(self, transform, temperature, scores):
        scores = np.array(list(scores.values()))
        if transform == "constant":
            weights = np.ones_like(scores)
        if transform == "max":
            weights = np.zeros_like(scores)
            scores = scores[:]
            for i in range(len(scores)):
                if i not in self.instance_scores.keys():
                    scores[i] = -float("inf")
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, max(scores))))
            weights[argmax] = 1.0
        elif transform == "eps_greedy":
            weights = np.zeros_like(scores)
            weights[np.argmax(scores)] = 1.0 - self.eps
            weights += self.eps / len(self.seeds)
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
