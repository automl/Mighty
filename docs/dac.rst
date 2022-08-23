Dynamic Algorithm Configuration with Mighty
===========================================

Dynamic Algorithm Configuration (DAC) [`Biedenkapp et al., ECAI 2020 <https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/20-ECAI-DAC.pdf>`_, `Adriaensen et al., CoRR 2022 <https://arxiv.org/pdf/2205.13881.pdf>`_]
is a hyperparameter optimization paradigm aiming to find the best possible hyperparameter configuration for a given *algorithm instance* at every *timestep* during runtime.
DAC can easily be modelled as a contextual MDP and is thus a real-world application of RL.

In order to interface with configurable algorithms, we recommend `DACBench <https://github.com/automl/DACBench>`_.
It provides algorithms from different fields as well as artificial benchmarks, all with the OpenAI gym interface.

In order to use Mighty with DACBench, you need to install DACBench first.
We recommend following the instructions in the DACBench repo.

Afterwards, select the benchmark you want to run, for example the SigmoidBenchmark, and providing it as the "env" keyword:

.. code-block:: bash

    python mighty/run_mighty.py 'algorithm=dqn' 'env=SigmoidBenchmark'


The benchmarks in DACBench have many configuration options. You can use your hydra configs to include your changes, simply use the keyword TDO

.. code-block:: bash

    python run_mighty.py TODO

Of course you can also load existing config files, e.g. to reproduce another experiment:

.. code-block:: bash

    python run_mighty.py TODO


