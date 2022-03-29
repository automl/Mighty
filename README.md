# Mighty-DACS

## Installation
We recommend to use the package manager like [anaconda or minconda](https://docs.anaconda.com/anaconda/install/)

## Setting up the python environment using Conda
If you have conda installed you can follow these steps to setup a clean python environment in which you can install the
needed packages. If you need to install conda [follow these steps](https://docs.anaconda.com/anaconda/install/).
The code has been tested with python 3.9.
First create a clean python environment

```bash
conda create --name mightydac python=3.9
conda activate mightydac
```

Then install the needed packages.

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -c=conda-forge
conda install ignite -c pytorch
pip install -r requirements.txt
```

Install Mighty

```bash
python setup.py develop
```

## Learning a Configuration Policy via DAC

TODO


## Run Coax Agent
```bash
python run_mighty.py 'num_steps=1000' '+algorithm=agent' 