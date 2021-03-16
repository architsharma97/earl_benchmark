# Wrapper around dm_control control suite environments
# that adds goal conditioning.
# Modeled after https://github.com/deepmind/dm_control/blob/master/dm_control/suite/wrappers/action_noise.py
# How to use: 
# import this_filename
# env = suite.load(...)
# env = this_filename.Wrapper(* your args here *)
# This file is still a WIP

import numpy as np
import dm_env
from dm_control.rl import control

class Wrapper(control.Environment):

    def __init__(self, env, goal, reward_func):
        self._env = env
        self._goal = goal
        self._reward_func = reward_func

    def reset(self):
        return self._env.reset()
        
    def step(self, action):
        time_step = self._env.step(action)
        # Replace reward with reward calculated from self._reward_func
        # self._reward_func takes in the environment and goal as parameters
        # and it is up to the caller to specify.
        time_step = dm_env.TimeStep(step_type = time_step.step_type, 
                                    reward = self._reward_func(self._env, self._goal),
                                    discount = time_step.discount,
                                    observation = time_step.observation) 
        return time_step

    def action_spec(self):
        return self._env.action_spec()

    def step_spec(self):
        return self._env.step_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    @property
    def physics(self):
        return self._env._physics

    @property
    def task(self):
        return self._env._task

    def control_timestep(self):
        return self._env.control_timestep()

