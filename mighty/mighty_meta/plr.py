import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent


class PrioritizedLevelReplay(MightyMetaComponent):
    def __init__(self, algo, env, alpha, rho, nu, staleness_coeff, sample_strategy, replay_schedule='proportional') -> None:
        super().__init__(algo)
        self.alpha = alpha
        self.rho = rho
        self.nu = nu
        self.staleness_coef = staleness_coeff
        self.sample_strategy = sample_strategy
        self.replay_schedule = replay_schedule
        self.instance_scores = {}
        self.staleness = {}
        self.all_instances = env.instance_set
        self.index = 0
        
        self.pre_episode_methods = [self.get_instance]
        self.post_episode_methods = [self.add_rollout]

    def get_instance(self, metrics=None):
        if self.sample_strategy == 'random':
            instance = np.random.choice(self.all_instances)
            return instance

        if self.sample_strategy == 'sequential':
            instance = self.all_instances[self.index]
            self.index = (self.index + 1) % len(self.all_instances)
            return instance

        num_unseen = len(self.all_instances)-len(list(self.instance_scores.keys()))
        proportion_seen = (len(self.all_instances) - num_unseen)/len(self.all_instances)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else: # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()
            
    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)

        idx = np.random.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[0]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance
    
    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.instance_scores)
        weights = weights * (1-self.unseen_seed_weights) # zero out unseen levels
        weights = [w for w in weights if i not in self.instance_scores.keys() else 0]

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.staleness)

            staleness_weights = [w for w in staleness_weights if i not in self.instance_scores.keys() else 0]
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights

    def _sample_unseen_level(self):
        sample_weights = np.zeros(len(self.all_instances))
        num_unseen = len(self.all_instances)-len(list(self.instance_scores.keys()))
        sample_weights = [1/num_unseen for i in self.all_instances if i not in self.instance_scores.keys() else 0]
        idx = np.random.choice(np.arange(len(self.all_instances)), 1, p=sample_weights)[0]
        instance = self.all_instances[idx]
        self._update_staleness(idx)
        return instance

    def _update_staleness(self, selected_id):
        if self.staleness_coef > 0:
            self.staleness = {k:v+1 for k,v in self.staleness.items()}
            self.staleness[selected_id] = 0

    def add_rollout(self, metrics):
        instance_id = metrics["instance_id"] 
        episode_reward = metrics["episode_reward"]  
        rollout_values = metrics["rollout_values"]  
        rollout_logits = metrics["rollout_logits"] 
        
        score = self.score_function(episode_reward, rollout_values, rollout_logits)
        if instance_id not in self.instance_scores.keys():
            old_score = 0
        else:
            old_score = self.instance_scores[instance_id]
        self.instance_scores[instance_id] = (1 - self.alpha)*old_score + self.alpha*score