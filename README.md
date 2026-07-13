<p align="center">
    <a href="./docs/img/logo.png">
        <img src="./docs/img/logo_with_font.svg" alt="Mighty Logo" width="80%"/>
    </a>
</p>

<div align="center">
    
[![PyPI Version](https://img.shields.io/pypi/v/mighty-rl.svg)](https://pypi.org/project/Mighty-RL/)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![License](https://img.shields.io/badge/License-BSD3-orange)
[![Test](https://github.com/automl/Mighty/actions/workflows/test.yaml/badge.svg)](https://github.com/automl/Mighty/actions/workflows/test.yaml)
[![Doc Status](https://github.com/automl/Mighty/actions/workflows/docs_test.yaml/badge.svg)](https://github.com/automl/Mighty/actions/workflows/docs_test.yaml)
    
</div>

<div align="center">
    <h3>
      <a href="#installation">Installation</a> |
      <a href="https://automl.github.io/Mighty/">Documentation</a> |
      <a href="#run-a-mighty-agent">Run a Mighty Agent</a> |
      <a href="#cite-us">Cite Us</a>
    </h3>
</div>

---

# Mighty

Welcome to Mighty, hopefully your future one-stop shop for everything cRL.
Currently Mighty is still in its early stages with support for normal gym envs, DACBench and CARL.
The interface is controlled through hydra and we provide DQN, PPO and SAC algorithms.
We log training and regular evaluations to file and optionally also to wandb.
If you have any questions or feedback, please tell us, ideally via the GitHub issues!
If you want to get started immediately, use our [Template repository](https://github.com/automl/mighty_project_template).

Mighty features:
- Modular structure for easy (Meta-)RL tinkering
- PPO, SAC and DQN as base algorithms
- Environment integrations via Gymnasium, Pufferlib, CARL & DACBench
- Implementations of some important baselines: RND, PLR, Cosine LR Schedule and more!

## Installation
We recommend to using uv to install and run Mighty in a virtual environment.
The code has been tested with python 3.11 on Unix systems.

First create a clean python environment:

```bash
uv venv --python=3.11
source .venv/bin/activate
```

Then  install Mighty:

```bash
make install
```

Optionally you can install the dev requirements directly:
```bash
make install-dev
```

Alternatively, you can install Mighty from PyPI:
```bash
pip install mighty-rl
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

### Train your Agent on a CARL Environment
Mighty is designed with contextual RL in mind and therefore fully compatible with CARL.
Before you start training, install the CARL extra so that `carl` is available in your
Mighty environment (see [CARL](https://github.com/automl/CARL) for more on the benchmark itself):
```bash
pip install -e ".[carl]"
```
(`make install-dev` installs this along with all other optional dependencies.)

Then use the same command as before, but provide the CARL environment, in this example CARLCartPoleEnv,
and information about the context distribution as keywords:
```bash
python mighty/run_mighty.py 'algorithm=ppo' 'env=CARLCartPole' '+env_kwargs.num_contexts=10' '+env_kwargs.context_feature_args.gravity=[normal, 9.8, 1.0, -100.0, 100.0]' 'env_wrappers=[mighty.mighty_utils.wrappers.FlattenVecObs]' 'algorithm_kwargs.rollout_buffer_kwargs.buffer_size=2048'
```

For more complex configurations like this, we recommend making an environment configuration file. Check out our [CARL Ant](mighty/configs/environment/carl_walkers/ant_goals.yaml) file to see how this simplifies the process of working with configurable environments.

### Learning a Configuration Policy via DAC

In order to use Mighty with DACBench, install the DACBench extra so that `dacbench`
is available in your Mighty environment (see [DACBench](https://github.com/automl/DACBench)
for more on the benchmark itself):
```bash
pip install -e ".[dacbench]"
```
(`make install-dev` installs this along with all other optional dependencies.)

Afterwards, configure the benchmark you want to run. Since most DACBench benchmarks have Dict action and observation spaces, some fairly complex,  you might need to wrap DACBenchmarks in order to translate the observations and actions to an easy-to-handle format. We have a version of the FunctionApproximationBenchmark configured for you so you can get started like this:
```bash
python mighty/run_mighty.py 'algorithm=ppo' 'environment=dacbench/function_approximation'
```
The matching [configuration file](mighty/configs/environment/dacbench/function_approximation.yaml) shows you how to set the search spaces and benchmark type. Refer to DACBench itself to learn how to configure other elements like observations spaces or instance sets.

### Optimize Hyperparameters
You can optimize the hyperparameters of your algorithm with the [Hypersweeper](https://github.com/automl/hypersweeper) package, e.g. using [SMAC3](https://github.com/automl/SMAC3). Mighty is directly compatible with Hypersweeper and thus smart and distributed HPO! There are also other HPO options, check out our [examples](examples/README.md) for more information.

## Build Your Own Mighty Project
If you want to implement your own method in Mighty, we recommend using the [Mighty template repository](https://github.com/automl/mighty_project_template) as a base. It contains a runscript, the most relevant config files and basic scripts for plotting. Our [domain randomization example](https://github.com/automl/mighty_dr_example) shows that you can get started right away. Since Mighty has many options of how to implement your idea, here's a rough guide which Mighty class you want to look at:

```mermaid
stateDiagram
  direction TB
  classDef Neutral stroke-width:1px,stroke-dasharray:none,stroke:#000000,fill:#FFFFFF,color:#000000;
  classDef Peach stroke-width:1px,stroke-dasharray:none,stroke:#FBB35A,fill:#FFEFDB,color:#8F632D;
  classDef Aqua stroke-width:1px,stroke-dasharray:none,stroke:#46EDC8,fill:#DEFFF8,color:#378E7A;
  classDef Sky stroke-width:1px,stroke-dasharray:none,stroke:#374D7C,fill:#E2EBFF,color:#374D7C;
  classDef Pine stroke-width:1px,stroke-dasharray:none,stroke:#254336,fill:#8faea5,color:#FFFFFF;
  classDef Rose stroke-width:1px,stroke-dasharray:none,stroke:#FF5978,fill:#FFDFE5,color:#8E2236;
  classDef Ash stroke-width:1px,stroke-dasharray:none,stroke:#999999,fill:#EEEEEE,color:#000000;
  classDef Seven fill:#E1BEE7,color:#D50000,stroke:#AA00FF;
  Still --> root_end:Yes
  Still --> Moving:No
  Moving --> Crash:Yes
  Moving --> s2:No, only current transitions, env and network
  s2 --> s6:Action Sampling
  s2 --> s10:Policy Update
  s2 --> s8:Training Batch Sampling
  s2 --> Crash:More than one/not listed
  s2 --> s12:Direct algorithm change
  s12 --> s13:Yes
  s12 --> s14:No
  Still:Modify training settings and then repeated runs?
  root_end:Runner
  Moving:Access to update infos (gradients, batches, etc.)?
  Crash:Meta Component
  s2:Which interaction point with the algorithm?
  s6:Exploration Policy
  s10:Update
  s8:Buffer
  s12:Change only the model architecture?
  s13:Network and/or Model
  s14:Agent
  class root_end Peach
  class Crash Aqua
  class s6 Sky
  class s8 Pine
  class s10 Rose
  class s13 Ash
  class s14 Seven
  class Still Neutral
  class Moving Neutral
  class s2 Neutral
  class s12 Neutral
  style root_end color:none
  style s8 color:#FFFFFF
```

## Pre-Implemented Methods
Mighty is meant to be a platform to build upon and not a large collection of methods in itself. We have a few relevant methods pre-implemented, however, and this collection will likely grow over time:

- **Agents**: SAC, PPO, DQN
- **Updates**: SAC, PPO, Q-learning, double Q-learning, clipped double Q-learning
- **Buffer**s: Rollout Buffer, Replay Buffer, Prioritized Replay Buffer
- **Exploration Policies**: e-greedy (with and without decay), ez-greedy, standard stochastic
- **Models** (with MLP, CNN or ResNet backbone): SAC, PPO, DQN (with soft and hard reset options)
- **Meta Components**: RND, NovelD, SPaCE, PLR
- **Runners**: online RL runner, ES runner

## Contributions

- **Aditya Mohan:** Implemented the core agent stack (training loop, model interfaces, evaluation tools), ran the experiments and evaluation, and maintained the codebase, including integration of new modules and cross-component consistency.
- **Theresa Eimer:** Designed the system architecture, Built the environment interfaces and data pipelines, contributed to architectural decisions, and implemented the meta-learning components and their evaluation setups.
- **Carolin Benjamins:** Reviewed the architecture, conducted focused code audits, and improved interface design and modular structure.
- **Marius Lindauer:** Guided the overall research direction, supervised the project, and refined the methodological setup.
- **André Biedenkapp:** Provided project supervision, contributed to research design, and advised on algorithmic methodology and experimentation.

## Cite Us

If you use Mighty in your work, please cite us:

```bibtex
@misc{mohaneimer24,
  author    = {A. Mohan and T. Eimer and C. Benjamins and M. Lindauer and A. Biedenkapp},
  title     = {Mighty},
  year      = {2024},
  url = {https://github.com/automl/mighty}
}
```
