Installation
============

We recommend to use the package manager like `anaconda or minconda <https://docs.anaconda.com/anaconda/install/>`_

If you have conda installed you can follow these steps to setup a clean python environment in which you can install the
needed packages. If you need to install conda `follow these steps <https://docs.anaconda.com/anaconda/install/>`_.
The code has been tested with python 3.9 only, so we highly recommend using this version as well.

First create a clean python environment:

.. code-block:: bash

    conda create --name mighty python=3.9
    conda activate mighty


Mac users should follow these additional installation steps in order to install all dependencies correctly.
First install jaxlib and jax via (as suggested in `this thread <https://github.com/google/jax/issues/5501#issuecomment-1032891169>`_)

.. code-block:: bash

    conda install -c conda-forge jaxlib
    conda install -c conda-forge jax

To install box2d-py you also need swig installed, which you can do via

.. code-block:: bash

    conda install swig


Then you can install Mighty itself:

.. code-block:: bash

    pip install .

