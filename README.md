# EARL: Environments for Autonomous Reinforcement Learning

EARL is an open-source benchmark for autonomous reinforcement learning, where the agent learns in a continual non-episodic setting without relying on extrinsic interventions for training. The benchmark consists of 6 challenging environments, covering diverse scenarios from dexterous manipulation to locomotion.

For an overview of the problem setting and description of the environments, checkout our [website](https://architsharma97.github.io/earl_benchmark/index.html). For more information on Autonomous RL and details about evaluation protocols and baselines, please refer to [conference publication]().

Current expected implementation:
```
import earl_benchmark

env_loader = earl_benchmark.EARLEnvs('tabletop_manipulation', reward_type='sparse')
train_env, eval_env = env_loader.get_envs()
initial_states = env_loader.get_initial_states()
goal_states = env_loader.get_goal_states()
forward_demos, reverse_demos = env_loader.get_demonstrations()

```