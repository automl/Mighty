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
The code has been tested with python 3.10.

First create a clean python environment:

```bash
conda create --name mighty python=3.10
conda activate mighty
```

:warning: **Due to the dependency on jaxlib/jax, MacOS is currently not supported.** :warning:



Then  install Mighty:

```bash
pip install .
```

## Run a Mighty Agent
In order to run a Mighty Agent, use the run_mighty.py script and provide any training options as keywords.
If you want to know more about the configuration options, call:
```bash
python mighty/run_mighty.py --help
```

An example for running the PPO agent on the Pendulum gym environment for 1000 steps looks like this:
```bash
python mighty/run_mighty.py 'num_steps=1000' 'algorithm=ppo' 'env=Pendulum-v1'
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
You could optimize the hyperparameters of your algorithm with the [hydra-smac-sweeper](https://github.com/automl/hydra-smac-sweeper) based on [SMAC3](https://github.com/automl/SMAC3).
After installing, you can use the hydra-smac-sweeper as follows:
1. Define the search space. For examples you can look at `mighty/configs/search_space/mighty_template.yaml`. It is important that you add `# @package hydra.sweeper.search_space` at the top of your search space config file such that it is inserted at the correct place.
2. Adjust your cluster in `mighty/configs/sweeper/smac.yaml`. If you want to use local SMAC you can specify `dask_client: null`.
3. Adjust the exact HPO method. Check the number of trials/function evaluations you want (`n_trials`). As default, we use a Gaussian Process with Expected Improvement via the BlackBoxFacade. You can have a look at the other [presets](https://automl.github.io/SMAC3/v2.0.1/3_getting_started.html#facade), i.e. for multi-fidelity, algorithm configuration and more.
4. Run HPO by adding `+sweeper=smac +search_space=mighty_template` to the commandline.

# In order to use this sweeper, add on the commandline:
# +sweeper=smac +search_space=mighty_template

## Further Examples
We provide further examples, such as how to plot the logged evaluation data, in the [examples](examples) folder.


## Optimizing on Cluster
### Noctua
```bash
ssh noctua2

# if not cloned
cd /scratch/hpc-prf-intexml/cbenjamins/repos/
git clone git@github.com:automl-private/Mighty-DACS.git

# if no env yet + not installed yet
micromamba create python=3.11 -p /scratch/hpc-prf-intexml/cbenjamins/envs/mighty -c conda-forge  # or conda
micromamba activate /scratch/hpc-prf-intexml/cbenjamins/envs/mighty
pip install -e .
# Install hydra smac sweeper

cd ..
git clone git@github.com:automl/hydra-smac-sweeper.git
cd hydra-smac-sweeper
pip install -e . --config-settings editable_mode=compat


# activate env
micromamba activate /scratch/hpc-prf-intexml/cbenjamins/envs/mighty

# sync runs
rsync -azv --delete -e 'ssh -J intexml2@fe.noctua2.pc2.uni-paderborn.de' intexml2@n2login5:/scratch/hpc-prf-intexml/cbenjamins/repos/Mighty-DACS/runs runs  # TODO update path



# create session if not active
tmux new -s attach

tmux attach
salloc --time=1:00:00 --nodes=1 --ntasks=8 --mem-per-cpu=4G
cd /scratch/hpc-prf-intexml/cbenjamins/repos/Mighty-DACS

python mighty/run_mighty.py -m env=CartPole-v1 +search_space=dqn_gym_classic algorithm=dqn 'num_steps=25e4' +cluster=noctua --config-name smac

nano /scratch/hpc-prf-intexml/cbenjamins/envs/mighty/lib/python3.11/site-packages/smac/runner/dask_runner.py

```
