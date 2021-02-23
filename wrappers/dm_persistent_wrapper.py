# Wrapper around dm_control control suite environments
# that adds Persistence.
# Modeled after https://github.com/deepmind/dm_control/blob/master/dm_control/suite/wrappers/action_noise.py
# How to use: 
# import this filename
# env = suite.load(...)
# env = this filename.Wrapper(* your args here *)
# This file is still a WIP

import numpy as np
import dm_env
from dm_control.rl import control

class Wrapper(control.Environment):

    def __init__(self, env, h):
        self._env = env
        self._h = h

    def reset(self):
        # TODO: return a value when reset is called but this condition is not true.
        # If condition is not true, then we should not reset. 
        # If we should not reset, what should this method return?
        if (self._env._step_count >= self._h or self._env._step_count == 0):
            return self._env.reset()
        
    def step(self, action):
        time_step = self._env.step(action)
        # Override reset_next_step if we are not at horizon yet,
        # or we are at horizon but episode is not over.
        if (self._env._reset_next_step and self._env._step_count < self._h):
            self._env._reset_next_step = False  
            time_step = dm_env.TimeStep(step_type = dm_env.StepType.MID, 
                                        reward = time_step.reward,
                                        discount = 1.0,
                                        observation = time_step.observation) 
        elif (self._env._step_count >= self._h and not self._env._reset_next_step):
            self._env._reset_next_step = True
            time_step = dm_env.TimeStep(step_type = dm_env.StepType.LAST, 
                                        reward = time_step.reward,
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

