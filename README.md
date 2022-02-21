# EARL: Environments for Autonomous Reinforcement Learning
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

EARL is an open-source benchmark for autonomous reinforcement learning, where the agent learns in a continual non-episodic setting without relying on extrinsic interventions for training. The benchmark consists of 6 challenging environments, covering diverse scenarios from dexterous manipulation to locomotion.

For an overview of the problem setting and description of the environments, checkout our [website](https://architsharma97.github.io/earl_benchmark/index.html). For more details on Autonomous RL and details about evaluation protocols and baselines, please refer to our [ICLR paper](https://arxiv.org/abs/2112.09605).

## Setup

EARL can be installed by cloning the repository as follows:
```
git clone https://github.com/architsharma97/earl_benchmark.git
cd earl_benchmark
conda env create -f env.yml
```

After installing the conda env, you can activate your environment with
```
conda activate earl
```

For the environments based on MuJoCo, you need to obtain a (free) [license](https://www.roboti.us/license.html) and copy the key into the subdirectory of your MuJoCo installation. 

## Using environments in EARL

You can load environments by first creating `earl_benchmark.EARLEnvs(env)`. You can then load initial and goal states and demonstrations (if available) as follows. 

```
import earl_benchmark

env_loader = earl_benchmark.EARLEnvs('tabletop_manipulation', reward_type='sparse')
train_env, eval_env = env_loader.get_envs()
initial_states = env_loader.get_initial_states()
goal_states = env_loader.get_goal_states()
forward_demos, reverse_demos = env_loader.get_demonstrations()
```

## Acknowledgements

EARL is built on top of environments built by various researchers. In particular, we would like to thank the authors of:
- [Meta-World](https://meta-world.github.io/) 
- [Relay Policy Learning](https://github.com/google-research/relay-policy-learning)
- [PyBullet](https://github.com/bulletphysics/bullet3)
- [MTRF](https://github.com/facebookresearch/MTRF)

## Disclaimer

The environment repository is WIP. Please contact [Archit Sharma](mailto:architsh@stanford.edu) if you are planning to use this benchmark and are having trouble.
