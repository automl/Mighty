import torch
from itertools import count
import numpy as np
from torch.autograd import Variable

from mighty.env.env_handling import DACENV


class EvaluationRolloutWorker(object):
    def __init__(
            self,
            policy,
            policy_type: str,
            device: str,
            env: DACENV
    ):
        self.policy = policy
        policy_types = ["continuous", "discrete"]
        if policy_type not in policy_types:
            raise ValueError(f"{policy_type} not available. Available policy types are {policy_types}.")
        self.policy_type = policy_type
        self.env = env
        self.device = device

    def eval(self, episodes: int = 1):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        It is expected that the max environment time steps are given by the environment.

        :returns (steps per episode), (reward per episode), (instance id per episode)
            If the environment is not a dacbench AbstractEnv, instance id will be -1.
        """
        steps, rewards = [], []
        instances = []
        with torch.no_grad():
            for e in range(episodes):
                es, er = 0, 0

                s = self.env.reset()
                if hasattr(self.env, "inst_id"):
                    instance_id = self.env.inst_id
                else:
                    instance_id = -1
                for _ in count():
                    a = self.get_action(state=s)
                    ns, r, d, _ = self.env.step(a)
                    er += r
                    es += 1
                    if d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                instances.append(instance_id)

        return steps, rewards, instances

    def get_action(self, state):
        if self.device == "cuda":
            state = Variable(torch.from_numpy(state).float().cuda(), requires_grad=False)
        else:
            state = Variable(torch.from_numpy(state).float(), requires_grad=False)

        if self.policy_type == 'discrete':
            action = np.argmax(self.policy(state).detach().numpy())
        elif self.policy_type == 'continuous':
            action = self.policy(state).detach().numpy()
        else:
            raise NotImplementedError

        return action
