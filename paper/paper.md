---
title: 'Mighty: A Comprehensive Tool for studying Generalization, Meta-RL and AutoRL'
tags:
  - Python
  - reinforcement learning
  - contextual reinforcement learning
  - meta-learning
  - generalization
  - automated machine learning
authors:
  - name: Aditya Mohan
    orcid: 0000-0003-0092-3780
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Theresa Eimer
    orcid: 0000-0001-5561-5908
    equal-contrib: true
    affiliation: 1
  - name: Carolin Benjamins
    orcid: 0009-0007-4643-3564
    affiliation: 1
  - name: Marius Lindauer
    orcid: 0000-0002-9675-3175
    affiliation: "1, 3"
  - name: André Biedenkapp
    orcid: 0000-0002-8703-8559
    affiliation: 2
affiliations:
 - name: Leibniz University Hannover, Germany
   index: 1
 - name: University of Freiburg, Germany
   index: 2
 - name: L3S Research Center, Germany
   index: 3
date: 5 December 2025
bibliography: paper.bib
repository: https://github.com/automl/Mighty
---

## Summary

Robust generalization, rapid adaptation, and automated tuning are critical for deploying reinforcement learning (RL) in real-world settings. Yet research in these areas remains fragmented across non-standard codebases and custom orchestration scripts. We introduce *Mighty*, an open-source library that unifies contextual generalization, Meta-RL, and AutoRL within a single modular interface. Mighty cleanly separates the *Agent* from a configurable environment modeled as a *Contextual MDP*, decoupling *inner-loop* updates from *outer-loop* adaptations. This enables unified support for: (i) contextual generalization and curriculum learning (e.g., unsupervised environment design), (ii) bi-level meta-learning (e.g., MAML, black-box strategies), and (iii) automated hyperparameter and architecture search (e.g. Bayesian optimization, evolutionary strategies, population‐based training). We outline Mighty's design and validate its implementation on standard RL benchmarks. By offering a unified modular platform, Mighty simplifies experimentation and accelerates research on robust, adaptable RL.

## Statement of need

Reinforcement learning (RL) has emerged as a powerful decision-making paradigm in complex and dynamic environments. Despite impressive successes in domains such as games [@silver-nature16a; @badia-icml20a; @vasco-rlc24] and robotics [@lee-sciro20a], RL algorithms frequently overfit their training conditions and struggle to generalize to new tasks [@benjamins-tmlr23a; @kirk-jair23a; @mohan-jair24a]. Addressing this challenge requires methods that not only learn efficiently on a single task but also adapt rapidly to novel settings and automatically tune their learning process.

Recent research has advanced in three complementary directions: (i) Generalization in RL [@benjamins-tmlr23a; @cho-neurips24a; @mohan-jair24a], (ii) Meta-RL methods [@kaushik-iros20a; @beck-fntml25a], and (iii) Automated RL (AutoRL) [@parkerholder-jair22a; @mohan-automlconf23a; @eimer-icml23a]. Although each has led to promising algorithms, researchers frequently resort to fragmented codebases and ad hoc scripting across environment design, RL training, and meta-optimization. This fragmentation increases engineering effort, impedes rapid iteration, and undermines reproducibility [@paradis-rlc].

We introduce *Mighty*: a modular library designed to enable research at the intersection of generalization, Meta-RL, and AutoRL. Mighty enforces a clean and principled separation between inner- and outer-loop processes, making it easy to combine, for example, curricula, context adaptation, and automated tuning within a unified framework. Users can prototype new methods, compose existing ones, and run controlled comparisons - all without ad hoc orchestration code.

![Overview of Mighty's concept and modules.](Figures/mighty_concept.pdf)

Mighty is designed around three design principles: *flexibility, smooth integration with existing libraries, and environment parallelization*. First, flexibility is central. Mighty exposes transitions, predictions, networks, and environments to meta-methods, enabling a broad range of research patterns including black-box outer loops, algorithm-informed inner loops, and environment-level interventions. Second, Mighty integrates smoothly with Gymnasium [@towers-arxiv24a], PufferLib [@suarez-rlc25], CARL [@benjamins-tmlr23a], and can interface with tools such as evosax [@evosax2022github] in under $100$ lines of code. This minimizes the glue code while preserving flexibility. Finally, Mighty uses standard Python and PyTorch for optimized networks with vectorized CPU environments for fast environment interaction. This design offers high training speeds, even for purely CPU-based environments, without sacrificing algorithmic modularity or code clarity.

## State of the field

The rapidly growing ecosystem of RL libraries spans diverse design philosophies -- from low-level composability [@weng-jmlr22a] to turnkey baselines [@raffin-jmlr21a; @huang-jmlr22a] and massive-scale engines [@toledo-misc24a] -- making direct comparison and tool selection challenging. Modular research frameworks expose the internal building blocks of an RL pipeline as standalone components that can be re-combined to quickly prototype new algorithms.  
TorchRL [@bou-arxiv23a] pioneered this approach in the PyTorch ecosystem, introducing the TensorDict abstraction to seamlessly pass the observations, actions and rewards between modules. Tianshou [@weng-jmlr22a] offers a similarly flexible design with separate *Policy*, *Collector*, and *Buffer* classes, enabling researchers to switch custom exploration strategies or data collection schemes with minimal boilerplate. Although these libraries excel at inner loop algorithm development and fine‐grained experimentation, counter to Mighty, they leave higher‐order workflows such as curriculum learning or meta-adaptation across tasks to external scripts or user‐written loops. Monolithic baselines such as Stable-baselines3 (SB3) [@raffin-jmlr21a] and CleanRL/PureJaxRL [@huang-jmlr22a; @lu-neurips22a] prioritize ease of use and reproducibility. However, this simplicity comes at the cost of extensibility: SB3's algorithms hide most of the training loop behind a single `learn()` call, and CleanRL's single file scripts are not designed for import or extension. Scalable platforms such as RLlib [@liang-icml18a; @liang-neurips21a] and STOIX [@toledo-misc24a] focus on maximizing throughput and supporting distributed execution. Although these systems shine when running large experiments, their APIs do not natively unify component modularity with built‐in meta-learning or curriculum design. 
Mighty occupies the middle ground, offering efficient single-node performance via PyTorch, straightforward multicore environment parallelism, and a modular interface within the same cohesive framework. 

## Software Design

Mighty is organized around three abstractions: (i) an Agent assembled from modular components (exploration, buffer, update rule, network parameterization), (ii) a Contextual MDP interface that treats environments as families parameterized by context, and (iii) a meta-layer split into runners (between-run orchestration such as HPO or population methods) and meta-components (within-run interventions via hook points). This separation keeps the training loop stable while enabling extension through Hydra configuration rather than editing core code.

**User Interface:** Mighty prioritizes usability and flexibility. We use Hydra [@yadan-github19a] for structured configuration files that expose all relevant training details without overwhelming new users. This also plugs Mighty into Hydra’s ecosystem for cluster execution and hyperparameter optimization. The algorithm components in Mighty are modular and can be replaced via configurations, allowing users to integrate new components without editing the training loop. *This keeps projects small, maintainable, and research-focused.* For example, to integrate domain randomization [@tobin-iros17] via Syllabus [@sullivan-rlj25], we need around $100$ lines of code each to interface Syllabus and build a custom task wrapper. With the [Mighty project template](https://github.com/automl/mighty_project_template) as a base, *less than $200$ lines of Python code and three configuration files* are enough for a full evaluation, including hyperparameter optimization and cluster deployment (see the [project repository](https://github.com/automl/mighty_dr_example/tree/main) including results).

**Agent Framework:** Mighty includes three base RL algorithms -- DQN [@mnih-nature15a], SAC [@haarnoja-icml18a] and PPO [@schulman-arxiv17a] -- built from four modular components: exploration policy, replay buffer, update function, and model parameterization. Each component is easily extendable, allowing users to swap in new methods without rewriting the entire algorithm or touching the training loop. Since these modules capture most of the algorithmic logic, this design supports a wide range of research. Our documentation features [an overview](https://automl.github.io/Mighty/package_structure/) on when and how to use each of Mighty's abstractions.

**Meta-Learning Framework:** Mighty's support for meta-methods is unique in the RL landscape. It offers two key abstractions: *runners* and *meta-components*. Runners control training lifecycles, interacting with agents and environments while accessing artifacts like performance metrics and policy weights. This supports use cases such as hyperparameter optimization, policy search with evolutionary methods (e.g., our evosax [@evosax2022github] runner), and more complex-to-implement Meta-RL algorithms like MAML [@finn-icml17a], which jointly adapts policy and environment. Meta-components operate within a single run, with access to six hook points and full training context. They can implement curriculum generation, intrinsic rewards, or dynamic hyperparameter schedules. Both runners and meta-components are modular, composable, and compatible across base agents.

**Currently Implemented Methods:** Mighty is primarily a platform to implement new research, but comes with several built-in options that demonstrate Mighty's functionality (a full overview can be found [in our documentation](https://automl.github.io/Mighty/)).
The $\epsilon$z-greedy [@dabney-iclr21] exploration, prioritized replay buffer [@schaul-iclr16a], and DDQN [@hasselt-aaai16a] update each expand upon the core agents.
In addition to our evosax runner, the meta-components show online interactions with hyperparameters (cosine annealing; [@loshchilov-iclr17a]), transitions (RND and NovelD; [@burda-iclr19a; @zhang-neurips21]) and contextual environments (PLR and SPaCE; [@jiang-icml21a; @eimer-icml21a]).

## Usage Example

```python
# Train a PPO agent on a contextual environment
python mighty/run_mighty.py 'algorithm=ppo' 'environment=carl/cartpole' \
    '+env_kwargs.num_contexts=10' \
    '+algorithm_kwargs.meta_methods=[mighty.mighty_meta.RND]'

# Run hyperparameter optimization with SMAC
python mighty/run_mighty.py --config-name=hypersweeper_smac_example_config -m
```

## Research Impact Statement

Mighty’s research contribution is a unified experimental substrate for studying generalization, Meta-RL, and AutoRL under consistent orchestration. By standardizing the interfaces for contextual environments and outer-loop optimization, Mighty reduces ad hoc scripting, improves comparability across methods, and supports reproducible ablations. Mighty is intended to accelerate research iteration rather than introduce a new RL algorithm or claim state-of-the-art performance by itself.
 

## Empirical Validation

We validate our implementations by comparing them with Open RL Benchmark results [@huang-arxiv24a]. Our aim is not to outperform existing baselines, but to demonstrate that Mighty achieves comparable performance at similar training budgets.
The following table reports the number of training steps, average wall clock time, and comparison of the final results between our implementations and the Open RL Benchmark reference values.

| Algorithm | Environment | Steps | Time (min) | Final Return | Open RL Benchmark Return |
|-----------|-------------|-------|------------|--------------|--------------------------|
| DQN | MountainCar | 5e5 | 51.1 | -200.00 ± 0.00 | -189.92 ± 11.00 |
| DQN | CartPole | 5e5 | 60.41 | 486.40 ± 30.77 | 499.92 ± 0.00 |
| PPO | MountainCar | 5e5 | 3.03 | -200.00 ± 0.00 | -200.00 ± 0.00 |
| PPO | CartPole | 5e5 | 3.67 | 479.80 ± 17.21 | 487.48 ± 6.79 |
| SAC | Walker2D | 1e6 | 353.13 | 4478.67 ± 689.22 | 4471.15 ± 1896.34 |
| SAC | HalfCheetah | 1e6 | 302.53 | 10588.34 ± 874.19 | 10958.60 ± 1335.62 |

The trends that broadly align are: PPO and DQN on CartPole closely track Open RL Benchmark, and PPO on MountainCar reproduces the expected $-200$ plateau. Deviations appear where exploration and continuous-control dynamics matter more: DQN on MountainCar remains at $-200$ on our runs while Open RL Benchmark occasionally escapes. SAC in Walker2D and HalfCheetah remains close to the mean performance reported by Open RL Benchmark, and within the variance of their performance across seeds. In general, the results demonstrate that Mighty's implementations reproduce the results of established baselines, both in sample efficiency and runtime.

## Acknowledgements

We acknowledge contributions from the AutoML community and thank the developers of CARL, DACBench, and other integrated frameworks that make Mighty's unified interface possible. Theresa Eimer, Aditya Mohan, and Marius Lindauer acknowledge funding by the German Research Foundation (DFG) under LI 2801/10-1. Carolin Benjamins and Marius Lindauer acknowledge funding by the German Research Foundation (DFG) under LI 2801/7-1. André Biedenkapp acknowledges funding through the research network “Responsive and Scalable Learning for Robots Assisting Humans” (ReScaLe) of the University of Freiburg. The ReScaLe project is funded by the Carl Zeiss Foundation.

## AI Usage Disclosure

We used large language model tools in a limited capacity for language editing (clarity, conciseness, and grammar) and, in the software repository, to assist with and co-author some commits (e.g., dependency and compatibility fixes), as reflected in the corresponding commit trailers. All technical claims, software design, experiments, and results were produced and verified by the authors, who take full responsibility for the paper and the code.

## References
