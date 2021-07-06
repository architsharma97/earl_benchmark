# Persistent RL Benchmark

Current expected implementation:
```
import persistent_rl_benchmark

env_loader = persistent_rl_benchmark.PersistentRLEnvs('tabletop_manipulation', reward_type='sparse', reset_train_env_at_goal=True)
train_env = env_loader.get_train_env()
eval_env = env_loader.get_eval_env()
initial_states = env_loader.get_initial_states()
goal_states = env_loader.get_goal_states()
demonstrations = env_loader.get_demonstrations() # not implemented yet

```