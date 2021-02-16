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
        if (self._step_count >= self._h):
            self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def action_spec(self):
        return self._env.action_spec()

    def step_spec(self):
        return self._env.step_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    @property
    def physics(self):
        return self._physics

    @property
    def task(self):
        return self._task

