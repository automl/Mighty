# General
import os
import numpy as np
from typing import Optional, Dict, Any

# Ignite
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint

# Mighty
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger
from mighty.utils.rollout_worker import RolloutWorker
from mighty.utils.extended_checkpointing import checkpoint_metadata

def print_epoch(engine):
    """
    Print epoch.

    TODO: delete this or replace by sophisticated logger.

    Parameters
    ----------
    engine: Engine
        ignite engine

    Returns
    -------
    None

    """
    episode = engine.state.epoch
    n_episodes = engine.state.max_epochs
    print("%s/%s" % (episode + 1, n_episodes))


class AbstractAgent:
    """
    Any mighty agent should implement this class
    """

    def __init__(
            self,
            env: DACENV,
            env_eval: DACENV,
            gamma: float,
            logger: Logger,
            output_dir: str,
            max_env_time_steps: int = 1_000_000,
            checkpoint_mode: str = 'latest'
    ):
        """
        Initialize an Agent
        :param gamma: discount factor
        :param env: environment to train on
        :param logger: data logger
        """
        self.gamma = gamma
        self.env = env
        self.logger = logger
        self.checkpoint_mode = checkpoint_mode
        #TODO: make util function that can detect this correctly for all kinds of gym spaces and use it here
        try:
            self._action_dim = self.env.action_space.n
        except AttributeError:
            self._action_dim = self.env.action_space.shape[0]
        self._state_shape = self.env.observation_space.shape[0]

        self._max_env_time_steps = max_env_time_steps  # type: int
        self._env_eval = env_eval

        self.output_dir = output_dir
        self.model_dir = os.path.join(self.output_dir, 'models')

        self.last_state = None
        self.total_steps = 0

        self._mapping_save_components = None  # type: Optional[Dict[str, Any]]

    def get_action(self, state: np.ndarray, epsilon: float):
        """
        Return action given a state and policy epsilon (NOTE: epsilon only makes sense in certain cases)
        :param state: environment state
        :param epsilon: epsilon for action selection
        :return: action for the next environment step
        """
        raise NotImplementedError

    def step(self):
        """
        Execute a single number of environment and training step
        """
        raise NotImplementedError

    def run_episode(self, episodes: int = 1):
        """
        Trains the agent for a given amount of episodes
        :param episodes: Training episodes
        """
        raise NotImplementedError

    def eval(self, env: DACENV, episodes: int = 1):
        """
        Evaluates the agent on a given environment
        :param env: evaluation environment
        :param episodes: number of evaluation episodes
        """
        raise NotImplementedError

    def checkpoint(self, filepath: str):
        """
        Save agent policy
        :param filepath: path to save point
        """
        raise NotImplementedError

    def load_agent(self, filepath: str):
        """
        Load agent from file
        :param filepath: path to agent
        """
        raise NotImplementedError

    def update_policy(self, deltas: np.ndarray):
        """
        Perform weight update with given deltas
        :param deltas: weight updates
        """
        raise NotImplementedError

    def check_termination(self, engine: Engine):
        """
        Check if an episode has completed.

        An episode is regarded as completed if the number of engine iteration exceeds self._max_env_time_steps. Fires
        the ignite event `Events.EPOCH_COMPLETED` if applicable.

        Parameters
        ----------
        engine: Engine
            Ignite engine.

        Returns
        -------
        None

        """
        if engine.state.iteration > self._max_env_time_steps:
            engine.fire_event(Events.EPOCH_COMPLETED)

    def start_episode(self, engine: Engine):
        """
        Start episode by resetting environment and logger.

        Parameters
        ----------
        engine: Engine
            Required signature for ignite engine.

        Returns
        -------
        None

        """
        self.last_state = self.env.reset()
        if self.logger is not None:
            self.logger.reset_episode()
            self.logger.set_train_env(self.env)

    def check_save_components(self):
        """
        Check if the mapping for saving components is set in the child class.

        Raises
        ------
        ValueError
            If self._mapping_save_components is not set in child class or is not a dictionary. Must set in child agent,
            e.g., {"actor": self.actor}.

        Returns
        -------
        None
        """
        if not self._mapping_save_components or type(self._mapping_save_components) is not dict:
            msg = "Please set '_mapping_save_components' in the child agent."
            raise ValueError(msg)

    def run_rollout(self, env: DACENV, episodes: int):
        """
        Run evaluation rollout.

        Parameters
        ----------
        env: DACEnv
            Environment to evaluate.
        episodes: int
            Number of episodes the environment should run.

        Returns
        -------
        None

        """
        # TODO: for this to be nice we want to separate policy and agent
        # agent = DDQN(self.env)
        # TODO: this should be easier
        for _, m in self.logger.module_logger.items():
            m.episode = self.logger.module_logger["train_performance"].episode
        f = None
        for fname in os.listdir(self.model_dir):
            if fname.startswith("eval_checkpoint"):
                f = fname
        if f is None:
            msg = "No suitable checkpoint for eval"
            raise ValueError(msg)
        worker = RolloutWorker(self, os.path.join(self.model_dir, f), self.logger)
        worker.evaluate(env=env, episodes=episodes)

    def load_checkpoint(self, path: str):
        msg = "Please implement loading from checkpoints in the child agent."
        raise NotImplementedError(msg)

    def train(
            self,
            n_episodes: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        """
        Train the agent.

        Parameters
        ----------
        n_episodes: int
            Number of training episodes.
        n_episodes_eval: int
            Number of evaluation episodes.
        eval_every_n_steps: int, optional
            Evaluate environment every n step/iterations. The default is every 1000 steps.
        human_log_every_n_episodes: int, optional
            Log progress for us humans every n episodes. The default is 100.
        save_model_every_n_episodes: int, optional
            Save model every n episodes. The default is 100. Additionally, the model will always be saved after the end
            of training.

        Returns
        -------
        None

        """
        # self._n_episodes_eval = n_episodes_eval
        self.check_save_components()

        # Init Engine
        trainer = Engine(self.step)

        # Register events
        # STARTED

        # EPOCH_STARTED
        # reset env
        trainer.add_event_handler(Events.EPOCH_STARTED, self.start_episode)

        # ITERATION_STARTED

        # ITERATION_COMPLETED
        eval_kwargs = dict(
            env=self._env_eval,
            episodes=n_episodes_eval,
        )
        eval_checkpoint_handler = ModelCheckpoint(self.model_dir, filename_prefix='eval', n_saved=None, create_dir=True)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=eval_every_n_steps), checkpoint_metadata, agent=self,
                                  file=f"{self.model_dir}/checkpoint_list.json", checkpoint_handler=eval_checkpoint_handler, engine=trainer)

        #trainer.add_event_handler(
        #    Events.ITERATION_COMPLETED(every=eval_every_n_steps),
        #    self.run_rollout,
        #    **eval_kwargs
        #)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.check_termination)

        # EPOCH_COMPLETED
        if self.checkpoint_mode is None:
            pass
        else:
            if self.checkpoint_mode == 'debug':
                n_saved = None
            elif self.checkpoint_mode == 'latest':
                n_saved = 1
            checkpoint_handler = ModelCheckpoint(self.model_dir, filename_prefix='', n_saved=n_saved, create_dir=True)
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=save_model_every_n_episodes), checkpoint_handler, to_save=self._mapping_save_components)
            if hasattr(self, '_replay_buffer'):
                if self.checkpoint_mode == 'debug':
                    self._replay_buffer.save(self.model_dir, self.total_steps)
                elif self.checkpoint_mode == 'latest':
                    if os.path.exists(os.path.join(self.model_dir, 'rpb.pkl')):
                        os.remove(os.path.join(self.model_dir, 'rpb.pkl'))
                    self._replay_buffer.save(self.model_dir)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=human_log_every_n_episodes), print_epoch)

        # COMPLETED
        # order of registering matters! first in, first out
        # we need to save the model first before evaluating
        trainer.add_event_handler(Events.COMPLETED, eval_checkpoint_handler, to_save=self._mapping_save_components)
        #trainer.add_event_handler(Events.COMPLETED, self.run_rollout, **eval_kwargs)

        # RUN
        iterations = range(self._max_env_time_steps)
        trainer.run(iterations, max_epochs=n_episodes)
