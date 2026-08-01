
Mighty is a Reinforcement Learning (RL) library that aims to make training general agents easy.
We natively support context in RL, i.e. train and test distributions that can be easily configured, as well
as Meta- and AutoRL methods on all levels.
That means if you're interested in general RL, you can start with well-known simulation environments and scale up
to actually applications using Mighty!

### What Can I Do With Mighty?
Mighty offers a lot of flexibility for training general agents with online RL:

- train on standard and contextual RL environments
- apply outer-loop methods like Bayesian Optimization or Evolutionary Strategies for Meta-Learning, Hyperparameter Optimization and more
- use in-the-loop ideas like curriculum learning to enhance training
- plug in modules for exploration, buffers or architectures without touching the full pipeline
- combine different methods for Meta- and AutoRL to form full RL pipelines

We currently do not support other learning paradigms, but might extend to e.g. include offline data as an option. 

### What is currently implemented in Mighty?
Mighty aims to provide a basic set of methods to demonstrate different usecases. This is what's currently implemented:

``` mermaid
%%{init: { "theme": "base", 
           "themeVariables": {"primaryColor": "#fffbfbff", "secondaryColor": "#fffeecff", "tertiaryColor": "#faffffff"}} }%%
mindmap
((Mighty))
  {{Buffers}}
    Rollout Buffer
    Replay Buffer
    Prioritized Replay Buffer
  {{Updates}}
    Q-Learning
      standard Q-Learning
      double Q-Learning
      clipped double Q-Learning
    PPO
    SAC
  {{Agents}}
    DQN
    PPO
    SAC
  {{Runners}}
    ES Runner
    online RL runner
  {{Meta Components}}
    Intrinsic Rewards
      NovelD
      RND
    Curricula
      SPaCE
      PLR
    Hyperparameters
      cosine annealing
  {{Exploration Policies}}
    e-greedy with optional decay
    ez-greedy
    standard stochastic policy
  {{Models with MLP, CNN or ResNet backbones}}
    DQN with soft and hard reset options
    SAC
    PPO
```

- **Agents**: SAC, PPO, DQN
- **Updates**: SAC, PPO, Q-learning, double Q-learning, clipped double Q-learning
- **Buffer**s: Rollout Buffer, Replay Buffer, Prioritized Replay Buffer
- **Exploration Policies**: e-greedy (with and without decay), ez-greedy, standard stochastic
- **Models** (with MLP, CNN or ResNet backbone): SAC, PPO, DQN (with soft and hard reset options)
- **Meta Components**: RND, NovelD, SPaCE, PLR
- **Runners**: online RL runner, ES runner

### Where Is Mighty Going?

Mighty currently supports DQN, PPO, and SAC with a range of contextual RL benchmarks and evaluation mechanisms. Future work will deepen support for Meta-RL and AutoRL.

### Contact & Citation
Mighty is developed at [LUHAI Hannover]() by members of [AutoRL.org](). Your first contact is lead maintainer [Aditya Mohan](). If you found issues or want to contribute new features, it's best to visit our [GitHub page](https://github.com/automl/Mighty) page and start a discussion.

If you use Mighty for your research, please cite our JOSS paper:

```bibtex
@article{mohan2026mighty,
  author    = {Aditya Mohan and Theresa Eimer and Carolin Benjamins and Marius Lindauer and Andr\'{e} Biedenkapp},
  title     = {Mighty: A Comprehensive Tool for Studying Generalization, Meta-RL and AutoRL},
  journal   = {Journal of Open Source Software},
  year      = {2026},
  volume    = {11},
  number    = {123},
  pages     = {10439},
  publisher = {The Open Journal},
  doi       = {10.21105/joss.10439},
  url       = {https://doi.org/10.21105/joss.10439}
}
```