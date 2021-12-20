"""Gym wrapper to return an environment for persistent state control.

TODO: we want to keep track of the number of resets, especially when the environment underneath can return done=True (irreversible set of environments)
"""

from gym import Wrapper

class PersistentStateWrapper(Wrapper):

  def __init__(self, env, episode_horizon):
    super(PersistentStateWrapper, self).__init__(env)
    self._episode_horizon = episode_horizon
    self._total_step_count = 0
    self._steps_since_reset = 0
    self._num_interventions = 0

  def reset(self):
    self._num_interventions += 1
    self._steps_since_reset = 0
    return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    
    self._total_step_count += 1
    self._steps_since_reset += 1

    if not done and self._steps_since_reset >= self._episode_horizon:
      done = True

    return obs, reward, done, info

  def is_successful(self, obs=None):
    if hasattr(self.env, 'is_successful'):
        return self.env.is_successful(obs)
    else:
        return False

  @property
  def num_interventions(self):
    return self._num_interventions

  @property
  def total_steps(self):
    return self._total_step_count
