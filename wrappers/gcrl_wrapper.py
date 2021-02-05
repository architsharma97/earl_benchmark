"""Gym wrapper environments to convert any environment to a goal-conditioned environment wrapper.

The environment retains its old state, but goals/tasks can be changed during the
episode.
"""

from gym import Wrapper

class GoalConditioningWrapper(Wrapper):

  def __init__(self, env):
    super(ResetFreeWrapper, self).__init__(env)
    

  def reset(self):
    pass

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    return obs, reward, done, info
