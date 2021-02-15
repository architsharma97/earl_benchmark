# Nikhil Sardana
# Wrapper around dm_control control suite environments
# that adds Goal Conditioning.
# Modeled after https://github.com/deepmind/dm_control/blob/master/dm_control/suite/wrappers/action_noise.py
# How to use: 
# import this filename
# env = suite.load(...)
# env = this filename.Wrapper(* your args here *)
# This file is still a WIP

import numpy as np
import dm_env

class Wrapper(dm_env.Environment):

    def __init__(self, env, goal, reward_func):
        self._env = env
        self._goal = goal
        self._reward_func = reward_func

    def change_goal(self, new_goal):
        self._goal = new_goal

    def step(self, action):
        time_step = self._env.step(action)
        if not time_step.reward == None:
            time_step.reward = self._reward_func(time_step.observation, self._goal)

