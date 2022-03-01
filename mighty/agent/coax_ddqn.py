import coax
import optax
import haiku as hk
import jax.numpy as jnp
from mighty.env.env_handling import DACENV
from mighty.utils.logger import Logger


class DDQNAgent(object):
    """
        Simple double DQN Agent
        """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tt(self, ndarray):
        """
        Helper Function to cast observation to correct type/device
        """
        if self.device == "cuda":
            return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
        else:
            return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

    def func_q(state, is_training):
        """ type-2 q-function: s -> q(s,.) """
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros)
        ))
        return seq(state)

    def __init__(
            self,
            env: DACENV,
            logger: Logger,
            eval_env: DACENV = None,
            lr: float = 0.01,
            epsilon: float = 0.1,
            batch_size: int = 64,
            render_progress: bool = True,
            log_tensorboard: bool = False
    ):
        self.lr = lr
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
            self.writer.add_scalar('hyperparameter/lr', self.lr)
            self.writer.add_scalar('hyperparameter/batch_size', self._batch_size)
            self.writer.add_scalar('hyperparameter/policy_epsilon', self._epsilon)

        self.initialize_agent()

    def initialize_agent(self):
        self.q = coax.Q(self.func_q, self.env)
        self.policy = coax.EpsilonGreedy(self.q, epsilon=self._epsilon)

        # target network
        self.q_target = self.q.copy()

        # specify how to update value function
        self.qlearning = coax.td_learning.DoubleQLearning(self.q, q_targ=self.q_target, optimizer=optax.adam(self.lr))

        # specify how to trace the transitions
        self.tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)
        print("Initialized agent.")

    def update_agent(self, step):
        transition_batch = self.buffer.sample(batch_size=self._batch_size)
        metrics_q = self.qlearning.update(transition_batch)
        # TODO: log these properly
        # env.record_metrics(metrics_q)

        # periodically sync target models
        if step % 10 == 0:
            self.q_target.soft_update(q, tau=1.0)

    def train(
            self,
            n_steps: int,
            n_episodes_eval: int,
            eval_every_n_steps: int = 1_000,
            human_log_every_n_episodes: int = 100,
            save_model_every_n_episodes: int = 100,
    ):
        self.logger.message("[DDQN] Learning")

        with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                'Remaining:',
                TimeRemainingColumn(),
                'Elapsed:',
                TimeElapsedColumn()
        ) as progress:
            steps = 0
            steps_since_eval = 0
            while steps < n_steps:
                s = self.env.reset()

                while not done:
                    a = self.policy(s)
                    s_next, r, done, info = self.env.step(a)
                    steps += 1

                    # add transition to buffer
                    self.tracer.add(s, a, r, done)
                    while tracer:
                        buffer.add(tracer.pop())

                    # update
                    self.update_agent(steps)
                    self.last_state = s
                    s = s_next

                if steps_since_eval >= eval_every_n_steps:
                    steps_since_eval = 0
                    self.eval(self.eval_env, n_episodes_eval)

            progress.finished = True  # TODO correctly emit progress finished
                # TODO add saving

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

    def load(self):
        """ Load checkpointed model. """
        raise NotImplementedError

    def eval(self, env, episodes):
        """
        Eval agent on an environment. (Full evaluation)
        :param env:
        :param episodes:
        :return:
        """
        raise NotImplementedError