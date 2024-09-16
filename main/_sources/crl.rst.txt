Contextual RL Environments in Mighty
====================================

Most RL environments are either not concerned with generalization at all or test generalization performance without providing much insight into what agents are tested on,
e.g. by using procedurally generated levels that are hard to understand as a structured training or test distribution.
Contextual RL (or cRL)[`Hallak et al., CoRR 2015 <https://arxiv.org/pdf/1502.02259.pdf>`_, `Benjamins et al., CoRR 2022 <https://arxiv.org/pdf/2202.04500.pdf>`_] aims to make the task distributions agents are trained on a specific as possible in order to gain better insights where agents perform well and
what is currently missing in RL generalization.

`CARL (context adaptive RL) <https://github.com/automl/CARL>`_ [`Benjamins et al., EcoRL 2021 <https://arxiv.org/pdf/2110.02102.pdf>`_] is a benchmark library specifically designed for contextual RL.
It provides highly configurable contextual extensions to several well-known RL environments and is what we recommend to get started in cRL.
Mighty is designed with contextual RL in mind and therefore fully compatible with CARL.
Before you start training, however, please follow the installation instructions in the `CARL repo <https://github.com/automl/CARL>`_.

Then use the same command as before, but provide the CARL environment, in this example CARLCartPoleEnv,
and information about the context distribution as keywords:

.. code-block:: bash

    python mighty/run_mighty.py 'algorithm=dqn' 'env=CARLCartPoleEnv' '+env_kwargs.num_contexts=10' '+env_kwargs.context_feature_args=[gravity]'
