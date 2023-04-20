import numpy as np
from mighty.mighty_meta.mighty_component import MightyMetaComponent

class SPaCE(MightyMetaComponent):
    def __init__(self, criterion='relative_improvement', threshold=0.1, k=1) -> None:
        super().__init__()
        self.criterion = criterion
        self.threshold = threshold
        self.instance_set = []
        self.k = k
        self.instance_set_size = k
        self.last_evals = None
        self.pre_episode_methods = [self.get_instances]

    def get_instances(self, metrics):
        env = metrics["env"]
        vf = metrics["vf"]
        rollout_values = None
        if "rollout_values" in metrics.keys():
            rollout_values = metrics["rollout_values"]
            
        if self.last_evals is None and rollout_values is None:
            self.all_instances = np.array(env.instance_id_list.copy())
            self.instance_set = self.all_instances[np.random.choice(self.all_instances, size=self.k)]
        elif self.last_evals is None:
            self.instance_set = self.all_instances[np.random.choice(self.all_instances, size=self.k)]
            self.last_evals = np.nanmean(rollout_values)
        else:
            if abs(np.mean(rollout_values) -self.last_evals)/(self.last_evals+1e-6) <= self.threshold:
                self.instance_set_size += self.k
            self.last_evals = np.nanmean(rollout_values)
            evals = self.get_evals(env, vf)
            #logging.info(evals)
            if self.criterion == 'improvement':
                improvement = (evals - self.last_evals) / self.last_evals
            elif self.criterion == 'relative_improvement':
                improvement = (evals - self.last_evals) / self.last_evals
            #logging.info(self.all_instances)
            #logging.info(np.argsort(improvement)[::-1])
            #logging.info(self.all_instances[np.argsort(improvement)[::-1]])
            self.instance_set = self.all_instances[np.argsort(improvement)[::-1]][:self.instance_set_size]
        env.instance_set = self.instance_set

    def get_evals(self, env, vf):
        values = []
        for i in self.all_instances:
            state, _ = env.reset(options={"instance_id": i})
            v = np.array(vf(state))
            # If we're dealing with a q function, we transform to value here
            if len(v) > 1:
                v = [sum(v)]
            values.append(v[0])
        return values
        