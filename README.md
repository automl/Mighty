# Mighty-DACS
Welcome to Mighty, hopefully your future one-stop shop for everything cRL.
Currently Mighty is still in its early stages with support for normal gym envs, DACBench and CARL.
The interface is controlled through hydra and we provide DQN, PPO and SAC algorithms.
We log training and regular evaluations to file and optionally also to tensorboard or wandb.
If you have any questions or feedback, please tell us, ideally via the GitHub issues!

## Installation
We recommend to use the package manager like [anaconda or minconda](https://docs.anaconda.com/anaconda/install/)

If you have conda installed you can follow these steps to setup a clean python environment in which you can install the
needed packages. If you need to install conda [follow these steps](https://docs.anaconda.com/anaconda/install/).
The code has been tested with python 3.9.

First create a clean python environment:

```bash
conda create --name mighty python=3.9
conda activate mighty
```

Then  install Mighty:

```bash
pip install .
```

## Run a Mighty Agent
In order to run a Mighty Agent, use the run_mighty.py script and provide any training options as keywords.
If you want to know more about the configuration options, call:
```bash
python run_mighty.py --help
```

An example for running the DQN agent on the Pendulum gym environment for 1000 steps looks like this:
```bash
python run_mighty.py 'num_steps=1000' 'algorithm=DQN' 'env=Pendulum-v1'
```

## Learning a Configuration Policy via DAC

In order to use Mighty with DACBench, you need to install DACBench first.
We recommend following the instructions in the [DACBench repo](https://github.com/automl/DACBench).

Afterwards, select the benchmark you want to run, for example the SigmoidBenchmark, and providing it as the "env" keyword: 
```bash
python run_mighty.py 'env=SigmoidBenchmark'
```

## Train your Agent on a CARL Environment
Mighty is designed with contextual RL in mind and therefore fully compatible with CARL.
Before you start training, however, please follow the installation instructions in the [CARL repo](https://github.com/automl/CARL).

Then use the same command as before, but provide the CARL environment, in this example CARLCartPoleEnv,
and information about the context distribution as keywords:
```bash
python run_mighty.py 'env=CARLCartPoleEnv' 'env_kwargs.num_contexts=10' 'env_kwargs.context_feature_args=[gravity]'
```