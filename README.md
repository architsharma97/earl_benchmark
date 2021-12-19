# EARL Benchmark

Current expected implementation:
```
import earl_benchmark

env_loader = earl_benchmark.EARLEnvs('tabletop_manipulation', reward_type='sparse', reset_train_env_at_goal=True)
train_env, eval_env = env_loader.get_envs()
initial_states = env_loader.get_initial_states()
goal_states = env_loader.get_goal_states()
forward_demos, reverse_demos = env_loader.get_demonstrations()

```

To specify task for the kitchen environment, use the kitchen_task argument when creating the env_loader (Currently supported tasks: open_microwave, bottom_burner, hinge_cabinet, light_switch).

For details on the environment, please look at the website:
<!-- add some environment figures here -->