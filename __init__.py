"API to load Persistent RL environments."

import os
import numpy as np
import pickle

from persistent_rl_benchmark.wrappers import persistent_state_wrapper

# for every environment, add an entry for the configuration of the environment
# make a default configuration for environment, the user can change the parameters by passing it to the constructor.

# number of initial states being provided to the user
# for deterministic initial state distributions, it should be 1
# for stochastic initial state distributions, sample the distribution randomly and save those samples for consistency
env_config = {
  'tabletop_manipulation': {
    'num_initial_state_samples': 1,
    'num_goals': 4,
    'train_horizon': int(2e5),
    'eval_horizon': 200,
  },
}

class PersistentRLEnvs(object):
  def __init__(self,
               # parameters that need to be set for every environment
               env_name,
               reward_type='sparse',
               reset_train_env_at_goal=False,
               # parameters that have default values in the config
               **kwargs):
    self._env_name = env_name
    self._reward_type = reward_type
    self._reset_train_env_at_goal = reset_train_env_at_goal

    # resolve to default parameters if not provided by the user
    self._train_horizon = kwargs.get('train_horizon', env_config[env_name]['train_horizon'])
    self._eval_horizon = kwargs.get('eval_horizon', env_config[env_name]['eval_horizon'])
    self._num_initial_state_samples = kwargs.get('num_initial_state_samples', env_config[env_name]['num_initial_state_samples'])

    self._train_env = self.get_train_env()
    self._eval_env = self.get_eval_env()

  def get_train_env(self):
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      train_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                             reward_type=self._reward_type,
                                                             reset_at_goal=self._reset_train_env_at_goal)

    return persistent_state_wrapper.PersistentStateWrapper(train_env, episode_horizon=self._train_horizon)

  def get_eval_env(self):
    if self._env_name == 'tabletop_manipulation':
      from persistent_rl_benchmark.envs import tabletop_manipulation
      eval_env = tabletop_manipulation.TabletopManipulation(task_list='rc_r-rc_k-rc_g-rc_b',
                                                            reward_type=self._reward_type)

    return persistent_state_wrapper.PersistentStateWrapper(eval_env, episode_horizon=self._eval_horizon)

  def get_envs(self):
    return self._train_env, self._eval_env

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

  def get_demonstrations(self):
    # use the current file to locate the demonstrations
    base_path = os.path.abspath(__file__)
    demo_dir = os.path.join(os.path.dirname(base_path), 'demonstrations')
    try:
      forward_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'forward/demo_data.pkl'), 'rb'))
      reverse_demos = pickle.load(open(os.path.join(demo_dir, self._env_name, 'reverse/demo_data.pkl'), 'rb'))
    except:
      print('please download the demonstrations corresponding to ', self._env_name)

    return forward_demos, reverse_demos
