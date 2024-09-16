Mighty on Standard Gym Environments
===================================

Mighty can be used as a standard RL library for all environments that follow the OpenAI gym interface.
In order to run a Mighty Agent, use the run_mighty.py script and provide any training options as keywords. If you want to know more about the configuration options, call:

.. code-block:: bash

    python mighty/run_mighty.py --help

An example for running the PPO agent on the Pendulum gym environment for 1000 steps looks like this:

.. code-block:: bash

    python mihgty/run_mighty.py 'num_steps=1000' 'algorithm=ppo' 'env=Pendulum-v1'