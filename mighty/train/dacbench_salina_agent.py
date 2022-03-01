import numpy as np
import torch

from salina.agents.gyma import AutoResetGymAgent


class AutoResetDACBenchAgent(AutoResetGymAgent):
    """
    Class that extends the AutoResetGymAgent to be useful with DACBench and has a notion of instances
    """
    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        input="action",
        output="env/",
        inst_split=None
    ):
        """
        make_env_fn (callable): A function that setsup a DACBench environment
        make_env_args (dict|list[dict]): Arguments that will be passed to make_env_fn
            If given as a single dict, all environments will be initialized with the same kwargs
            If given as a list, environments can be individually configured
        n_envs (int): Number of environments to spawn
        input (str): Name of element to query from the workspace (default='action')
        output (str): Prefix under which new elements are added to the workspace (default='env/')
        inst_split (None|str): Decides how to split up instances across environments. Can either be None, 'equal' or
            'equal-seed'.
            None -> Don't split instances across enviornments and only change the seed per environment
            'equal' -> Keep fixed seed but split instances across environments in equal chunks
            'equal-seed' -> Change seed per environment and split instances equally across environments
        """
        super().__init__(make_env_fn, make_env_args, n_envs, input, output)
        self._inst_split = inst_split

    def _initialize_envs(self, n):
        assert self._seed is not None, "[GymAgent] seeds must be specified"
        if isinstance(self.env_args, dict):
            self.envs = [self.make_env_fn(**self.env_args) for _ in range(n)]

        elif isinstance(self.env_args, list):
            assert len(self.env_args) == n,\
                '[GymAgent] provide the same number of configurations as requested environments'
            self.envs = [self.make_env_fn(**self.env_args[k]) for k in range(n)]

        # Split instances across environments
        if self._inst_split is None:
            for k in range(n):
                self.envs[k].seed(self._seed + k)
        elif self._inst_split == 'equal':  # Same seeds different insts with equal sized splits
            inst_splits = np.array_split(self.envs[0].instance_id_list, n)
            for k in range(n):
                self.envs[k].seed(self._seed)
                self.envs[k].instance_id_list = inst_splits[k]
                self.envs[k].instance_index = 0
                self.envs[k].inst_id = inst_splits[k][0]
                self.envs[k].instance = self.envs[k].instance_set[self.envs[k].inst_id]
        elif self._inst_split == 'equal-seed':  # Different seeds and different insts splits of equal size
            inst_splits = np.array_split(self.envs[0].instance_id_list, n)
            for k in range(n):
                self.envs[k].seed(self._seed + k)
                self.envs[k].instance_id_list = inst_splits[k]
                self.envs[k].instance_index = 0
                self.envs[k].inst_id = inst_splits[k][0]
                self.envs[k].instance = self.envs[k].instance_set[self.envs[k].inst_id]
        else:
            raise NotImplementedError

        self.n_envs = n
        self.timestep = 0
        self.finished = torch.tensor([True for _ in self.envs])
        self.timestep = torch.tensor([0 for _ in self.envs])
        self.is_running = [False for _ in range(n)]
        self.cumulated_reward = {}

    def _reset(self, k, save_render):
        ret = super()._reset(k, save_render)
        ret['inst'] = torch.tensor([self.envs[k].inst_id]).int()
        # Make sure the instance that corresponds to the current state is logged to the workspace
        return ret

    def _step(self, k, action, save_render):
        ret = super()._step(k, action, save_render)
        ret['inst'] = torch.tensor([self.envs[k].inst_id]).int()
        # Make sure the instance that corresponds to the current state is logged to the workspace
        return ret