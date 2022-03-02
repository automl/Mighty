import os
import jax
import coax
import optax
import haiku as hk
import jax.numpy as jnp
from rich.progress import Progress, TimeRemainingColumn, TimeElapsedColumn, BarColumn

from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger


class MightyAgent(object):
    """
    Simple double DQN Agent
    """
    def __init__(
            self,
            env: DACENV,
            logger: Logger,
            eval_env: DACENV = None,
            learning_rate: float = 0.01,
            epsilon: float = 0.1,
            batch_size: int = 64,
            render_progress: bool = True,
            log_tensorboard: bool = False
    ):
        self.learning_rate = learning_rate
        self._epsilon = epsilon
        self._batch_size = batch_size

        if logger is not None:
            output_dir = logger.log_dir
        else:
            output_dir = None

        self.env = env
        if eval_env is None:
            self.eval_env = self.env
        else:
            self.eval_env = eval_env

        self.logger = logger
        self.render_progress = render_progress
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.model_dir = os.path.join(self.output_dir, 'models')

        self.last_state = None
        self.total_steps = 0

        self.writer = None
        if log_tensorboard and output_dir is not None:
            self.writer = SummaryWriter(output_dir)
            self.writer.add_scalar('hyperparameter/learning_rate', self.learning_rate)
            self.writer.add_scalar('hyperparameter/batch_size', self._batch_size)
            self.writer.add_scalar('hyperparameter/policy_epsilon', self._epsilon)

        self.initialize_agent()

    def initialize_agent(self):
        raise NotImplementedError

    def update_agent(self, step):
        raise NotImplementedError

    def train(
            self,
            n_steps: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        step_progress = 1 / n_steps
        episodes = 0
        with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                'Remaining:',
                TimeRemainingColumn(),
                'Elapsed:',
                TimeElapsedColumn()
        ) as progress:
            steps_task = progress.add_task("Train Steps", total=n_steps, start=False, visible=False)
            progress.start_task(steps_task)
            steps = 0
            steps_since_eval = 0
            log_reward_buffer = []
            while steps < n_steps:
                progress.update(steps_task, visible=True)
                s = self.env.reset()
                done = False
                while not done:
                    a = self.policy(s)
                    s_next, r, done, info = self.env.step(a)
                    log_reward_buffer.append(r)
                    steps += 1
                    progress.advance(steps_task)

                    # add transition to buffer
                    self.tracer.add(s, a, r, done)
                    while self.tracer:
                        self.buffer.add(self.tracer.pop())

                    # update
                    if len(self.buffer) >= self._batch_size:
                        self.update_agent(steps)

                    self.last_state = s
                    s = s_next
                    
                episodes += 1

                if steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.eval(self.eval_env, n_episodes_eval)

                # TODO: make this more informative
                if episodes % human_log_every_n_episodes == 0:
                    print(f"Steps: {steps}, Reward: {sum(log_reward_buffer) / len(log_reward_buffer)}")
                    log_reward_buffer = []

                if episodes % save_model_every_n_episodes == 0:
                    self.save()

    def run(
            self,
            n_steps: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        self.train(
            n_steps=n_steps,
            n_episodes_eval=n_episodes_eval,
            eval_every_n_steps=eval_every_n_steps,
            human_log_every_n_episodes=human_log_every_n_episodes,
            save_model_every_n_episodes=save_model_every_n_episodes
        )

    def load(self, path):
        """ Load checkpointed model. """
        raise NotImplementedError

    def save(self):
        """ Checkpoint model. """
        raise NotImplementedError

    def eval(self, env, episodes):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError
