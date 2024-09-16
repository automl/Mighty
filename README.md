<p align="center">
    <a href="./docs/img/logo.png">
        <img src="./docs/img/logo.png" alt="Mighty Logo" width="80%"/>
    </a>
</p>

<div align="center">
    
[![PyPI Version](https://img.shields.io/pypi/v/mighty-rl.svg)](https://pypi.python.org/pypi/mighty)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![License](https://img.shields.io/badge/License-BSD3-orange)
[![Test](https://github.com/automl/mighty/actions/workflows/pytest.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/test.yaml)
[![Doc Status](https://github.com/automl/mighty/actions/workflows/docs.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/docs.yaml)
    
</div>

<div align="center">
    <h3>
      <a href="#installation">Installation</a> |
      <a href="#quickstart">Run a Mighty Agent</a> |
      <a href="#cite-us">Cite Us</a>
    </h3>
</div>

---

# Mighty

**Warning: Mighty is still in development without an official release! Use at your own peril and check back frequently for updates!**

Welcome to Mighty, hopefully your future one-stop shop for everything cRL.
Currently Mighty is still in its early stages with support for normal gym envs, DACBench and CARL.
The interface is controlled through hydra and we provide DQN, PPO and SAC algorithms.
We log training and regular evaluations to file and optionally also to tensorboard or wandb.
If you have any questions or feedback, please tell us, ideally via the GitHub issues!

Mighty features:
- Modular structure for easy (Meta-)RL tinkering
- PPO, SAC and DQN as base algorithms
- Environment integrations via Gymnasium, Pufferlib, CARL & DACBench
- Implementations of some important baselines: MAML, PLR, Cosine LR Schedule and more!

## Installation
We recommend to use the package manager like [anaconda or minconda](https://docs.anaconda.com/anaconda/install/)

If you have conda installed you can follow these steps to setup a clean python environment in which you can install the
needed packages. If you need to install conda [follow these steps](https://docs.anaconda.com/anaconda/install/).
The code has been tested with python 3.10.

First create a clean python environment:

```bash
conda create --name mighty python=3.10
conda activate mighty
```

Then  install Mighty:

```bash
make install
```

## Run a Mighty Agent
In order to run a Mighty Agent, use the run_mighty.py script and provide any training options as keywords.
If you want to know more about the configuration options, call:
```bash
python mighty/run_mighty.py --help
```

An example for running the PPO agent on the Pendulum gym environment looks like this:
```bash
python mighty/run_mighty.py 'algorithm=ppo' 'environment=gymnasium/pendulum'
```

## Learning a Configuration Policy via DAC

In order to use Mighty with DACBench, you need to install DACBench first.
We recommend following the instructions in the [DACBench repo](https://github.com/automl/DACBench).

Afterwards, select the benchmark you want to run, for example the SigmoidBenchmark, and providing it as the "env" keyword: 
```bash
python mighty/run_mighty.py 'algorithm=dqn' 'env=SigmoidBenchmark' 'env_wrappers=[dacbench.wrappers.MultiDiscreteActionWrapper]'
```

## Train your Agent on a CARL Environment
Mighty is designed with contextual RL in mind and therefore fully compatible with CARL.
Before you start training, however, please follow the installation instructions in the [CARL repo](https://github.com/automl/CARL).

Then use the same command as before, but provide the CARL environment, in this example CARLCartPoleEnv,
and information about the context distribution as keywords:
```bash
python mighty/run_mighty.py 'algorithm=dqn' 'env=CARLCartPoleEnv' '+env_kwargs.num_contexts=10' '+env_kwargs.context_feature_args=[gravity]'
```

## Optimize Hyperparameters
You can optimize the hyperparameters of your algorithm with the [Hypersweeper](https://github.com/automl/hypersweeper) package, e.g. using [SMAC3](https://github.com/automl/SMAC3). Mighty is directly compatible with Hypersweeper and thus smart and distributed HPO!

## Further Examples
We provide further examples, such as how to plot the logged evaluation data, in the [examples](examples) folder.

## Cite Us

If you use Mighty in your work, please cite us:

```bibtex
@misc{mohaneimer24,
  author    = {A. Mohan and T. Eimer and C. Benjamins and F. Hutter and M. Lindauer and A. Biedenkapp},
  title     = {Mighty},
  year      = {2024},
  url = {https://github.com/automl/mighty},
```
