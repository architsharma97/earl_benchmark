"API to load Persistent RL environments."

import os
import numpy as np
import pickle

from persistent_rl_benchmark.wrappers import persistent_state_wrapper
from persistent_rl_benchmark.wrappers import lifelong_wrapper

# for every environment, add an entry for the configuration of the environment
# make a default configuration for environment, the user can change the parameters by passing it to the constructor.

# number of initial states being provided to the user
# for deterministic initial state distributions, it should be 1
# for stochastic initial state distributions, sample the distribution randomly and save those samples for consistency
transfer_env_config = {
  'tabletop_manipulation': {
    'num_initial_state_samples': 1,
    'num_goals': 4,
    'train_horizon': int(2e5),
    'eval_horizon': 200,
  },
  'tabletop_3obj': {
    'num_initial_state_samples': 1,
    'num_goals': 4,
    'train_horizon': int(2e5),
    'eval_horizon': 200,
  },
  'sawyer_door': {
    'num_initial_state_samples': 1,
    'num_goals': 1,
    'train_horizon': int(2e5),
    'eval_horizon': 300,
  },
  'sawyer_peg': {
    'num_initial_state_samples': 15,
    'num_goals': 1,
    'train_horizon': int(1e5),
    'eval_horizon': 200,
  },
  'kitchen': {
    'num_initial_state_samples': 1,
    'train_horizon': int(2e5),
    'eval_horizon': 200,
    'task': 'all_pairs',
  },
}

# for lifelong versions of the problem, only set the training horizons and goal/task change frequency.
lifelong_env_config = {
  'tabletop_manipulation': {
    'num_initial_state_samples': 1,
    'num_goals': 4,
    'train_horizon': int(5e4),
    'goal_change_frequency': 400,
  },
  'sawyer_door': {
    'num_initial_state_samples': 1,
    'num_goals': 1,
    'train_horizon': int(5e4),
    'goal_change_frequency': 600,
  },
  'sawyer_peg': {
    'num_initial_state_samples': 15,
    'num_goals': 1,
    'train_horizon': int(5e4),
    'goal_change_frequency': 400,
  },
  'kitchen': {
    'num_initial_state_samples': 1,
    'train_horizon': int(5e4),
    'goal_change_frquency': 400,
    'task': 'all_pairs',
  },
}

class PersistentRLEnvs(object):
  def __init__(self,
               # parameters that need to be set for every environment
               env_name,
               reward_type='sparse',
               reset_train_env_at_goal=False,
               setup_as_lifelong_learning=False,
               # parameters that have default values in the config
               **kwargs):
    self._env_name = env_name
    self._reward_type = reward_type
    self._reset_train_env_at_goal = reset_train_env_at_goal
    self._setup_as_lifelong_learning = setup_as_lifelong_learning
    self._kwargs = kwargs

    # resolve to default parameters if not provided by the user
    if not self._setup_as_lifelong_learning:
      self._train_horizon = kwargs.get('train_horizon', transfer_env_config[env_name]['train_horizon'])
      self._eval_horizon = kwargs.get('eval_horizon', transfer_env_config[env_name]['eval_horizon'])
      self._num_initial_state_samples = kwargs.get('num_initial_state_samples', transfer_env_config[env_name]['num_initial_state_samples'])

      self._train_env = self.get_train_env()
      self._eval_env = self.get_eval_env()
    else:
      self._train_horizon = kwargs.get('train_horizon', lifelong_env_config[env_name]['train_horizon'])
      self._num_initial_state_samples = kwargs.get('num_initial_state_samples', lifelong_env_config[env_name]['num_initial_state_samples'])
      self._goal_change_frequency = kwargs.get('goal_change_frequency', lifelong_env_config[env_name]['goal_change_frequency'])
      self._train_env = self.get_train_env(lifelong=True)

  def get_train_env(self, lifelong=False):
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      train_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                             reward_type=self._reward_type,
                                                             reset_at_goal=self._reset_train_env_at_goal)

    elif self._env_name == 'tabletop_3obj':
      from persistent_rl_benchmark.envs import tabletop_manipulation_3obj
      train_env = tabletop_manipulation_3obj.TabletopManipulation(reward_type=self._reward_type,
                                                                  reset_at_goal=self._reset_train_env_at_goal)
    elif self._env_name == 'sawyer_door':
      from persistent_rl_benchmark.envs import sawyer_door
      train_env = sawyer_door.SawyerDoorV2(reward_type=self._reward_type,
                                           reset_at_goal=self._reset_train_env_at_goal)
    elif self._env_name == 'sawyer_peg':
      from persistent_rl_benchmark.envs import sawyer_peg
      train_env = sawyer_peg.SawyerPegV2(reward_type=self._reward_type,
                                         reset_at_goal=self._reset_train_env_at_goal)
    elif self._env_name == 'kitchen':
      from persistent_rl_benchmark.envs import kitchen
      kitchen_task = self._kwargs.get('kitchen_task', transfer_env_config[self._env_name]['task'])  
      train_env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type)

    train_env = persistent_state_wrapper.PersistentStateWrapper(train_env, episode_horizon=self._train_horizon)
    
    if not lifelong:
      return train_env
    else:
      return lifelong_wrapper.LifelongWrapper(train_env, self._goal_change_frequency)

  def get_eval_env(self):
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      eval_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                            reward_type=self._reward_type)
    elif self._env_name == 'sawyer_door':
      from persistent_rl_benchmark.envs import sawyer_door
      eval_env = sawyer_door.SawyerDoorV2(reward_type=self._reward_type)
    elif self._env_name == 'sawyer_peg':
      from persistent_rl_benchmark.envs import sawyer_peg
      eval_env = sawyer_peg.SawyerPegV2(reward_type=self._reward_type)
    elif self._env_name == 'tabletop_3obj':
      from persistent_rl_benchmark.envs import tabletop_manipulation_3obj
      eval_env = tabletop_manipulation_3obj.TabletopManipulation(reward_type=self._reward_type)
    elif self._env_name == 'kitchen':
      from persistent_rl_benchmark.envs import kitchen
      kitchen_task = self._kwargs.get('kitchen_task', transfer_env_config[self._env_name]['task'])  
      eval_env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type)


    return persistent_state_wrapper.PersistentStateWrapper(eval_env, episode_horizon=self._eval_horizon)

  def get_envs(self):
    if not self._setup_as_lifelong_learning:
      return self._train_env, self._eval_env
    else:
      return self._train_env

  def get_initial_states(self, num_samples=None):
    '''
    Always returns initial state of the shape N x state_dim
    '''
    if num_samples is None:
      num_samples = self._num_initial_state_samples

    # TODO: potentially load initial states from disk
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      return tabletop_manipulation.initial_states

    elif self._env_name == 'sawyer_door':
      from persistent_rl_benchmark.envs import sawyer_door
      return sawyer_door.initial_states

    elif self._env_name == 'sawyer_peg':
      from persistent_rl_benchmark.envs import sawyer_peg
      return sawyer_peg.initial_states

    elif self._env_name == 'tabletop_3obj':
      from persistent_rl_benchmark.envs import tabletop_manipulation_3obj
      return tabletop_manipulation_3obj.initial_states

    elif self._env_name == 'kitchen':
      from persistent_rl_benchmark.envs import kitchen
      kitchen_task = self._kwargs.get('kitchen_task', transfer_env_config[self._env_name]['task'])  
      env = kitchen.Kitchen(task=kitchen_task, reward_type=self._reward_type)
      return env.get_init_states()

    else:
      # make a new copy of environment to ensure that related parameters do not get affected by collection of reset states
      cur_env = self.get_eval_env()
      reset_states = []
      while len(reset_states) < self._num_initial_state_samples:
        reset_states.append(cur_env.reset())
        reset_states = list(set(reset_states))

      return np.stack(reset_states)

  def get_goal_states(self):
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      return tabletop_manipulation.goal_states

    elif self._env_name == 'sawyer_door':
      from persistent_rl_benchmark.envs import sawyer_door
      return sawyer_door.goal_states
    
    elif self._env_name == 'sawyer_peg':
      from persistent_rl_benchmark.envs import sawyer_peg
      return sawyer_peg.goal_states

    elif self._env_name == 'tabletop_3obj':
      from persistent_rl_benchmark.envs import tabletop_manipulation_3obj
      return tabletop_manipulation_3obj.goal_states

    if self._env_name == 'kitchen':
      from persistent_rl_benchmark.envs import kitchen
      return kitchen.goal_states

  def get_demonstrations(self):
    # use the current file to locate the demonstrations
    base_path = os.path.abspath(__file__)
    demo_dir = os.path.join(os.path.dirname(base_path), 'demonstrations')
    try:
      forward_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'forward/demo_data.pkl'), 'rb'))
      reverse_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'reverse/demo_data.pkl'), 'rb'))
      return forward_demos, reverse_demos
    except:
      print('please download the demonstrations corresponding to ', self._env_name)
