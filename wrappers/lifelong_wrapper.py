"""Gym wrapper to setup a lifelong learning task.

TODO: Setup alternate task setting routines beyond random goal sampling at fixed intervals.
"""

from gym import Wrapper

class LifelongWrapper(Wrapper):
  '''
  A wrapper to setup the lifelong learning problem in a given goal-conditioned environment.
  The wrapper monitors the entire return over the course of its lifetime.

  For multi-goal environments, the wrapper changes to a random goal after a fixed number of steps.

  TODO: The wrapper will not be able to monitor the lifelong reward correctly if the user changes
  the environment goal (especially in the transfer setting).
  '''
  def __init__(self, env, goal_change_frequency):
    super(LifelongWrapper, self).__init__(env)
    self._num_interventions = 0
    self._lifelong_return = 0
    self._steps_since_goal_change = 0
    self._goal_change_frequency = goal_change_frequency

  def reset(self):
    self._num_interventions += 1
    self._steps_since_goal_change = 0
    return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._steps_since_goal_change += 1
    self._lifelong_return += reward

    if self._steps_since_goal_change >= self._goal_change_frequency:
      self._steps_since_goal_change = 0
      self.env.reset_goal() # assumes that env reset_goal function for randomly sampling the goal

      try:
          obs = self.env._get_obs() # assumes a _get_obs method (with modified goal)
      except:
          obs = self.env.env._get_obs() # assumes a _get_obs method (with modified goal)


    return obs, reward, done, info

  @property
  def lifelong_return(self):
    return self._lifelong_return

  @property
  def num_interventions(self):
    return self._num_interventions
