""""Kitchen environment."""

# Add Kitchen assets adept_envs/ folder to the python path.
import sys
import os
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(parent_dir, "kitchen_assets/adept_envs"))

import time
import numpy as np
import mujoco_py
import copy
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

component_to_state_idx = {
    'arm': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'burner0': [9, 10],
    'burner1': [11, 12],
    'burner2': [13, 14],
    'burner3': [15, 16],
    'light_switch': [17, 18],
    'slide_cabinet': [19],
    'hinge_cabinet': [20, 21],
    'microwave': [22],
}

# clean goal state
goal_states = np.array([[
  -4.1336253e-01,
  -1.6970085e+00,
   1.4286385e+00,
  -2.5005307e+00,
   6.2198675e-01,
   1.2632011e+00,
   8.8903642e-01,
   4.3514766e-02,
   7.9217982e-03,
  -5.1586074e-04,
   4.8548312e-04,
  -5.4527864e-06,
   6.3510129e-06,
   6.0837720e-05,
  -3.3861103e-05,
   6.6394619e-05,
  -1.9801613e-05,
  -1.2477605e-04,
   3.8065159e-04,
  -1.5148541e-04,
  -9.2229841e-04,
   7.2293887e-03,
   6.9650509e-03,
]])

# supported_tasks = ['close_microwave', 'burner0', 'burner1', 'burner2', 'burner3', 'light_switch', 'slide_cabinet', 'hinge_cabinet']
shaped_reward_tasks = ['microwave', 'light_switch', 'slide_cabinet', 'hinge_cabinet']

initial_states = {}

def convert_to_initial_state(component_names, values):
  new_init_state = goal_states[0].copy()
  for name, val in zip(component_names, values):
    new_init_state[component_to_state_idx[name]] = np.array(val)
  return new_init_state

# states from https://github.com/rail-berkeley/d4rl/blob/master/d4rl/kitchen/kitchen_envs.py
initial_states['microwave'] = convert_to_initial_state(['microwave'], [[-0.7]])
# initial_states['burner1'] = convert_to_initial_state('burner1', [-0.88, -0.01])
# initial_states['burner3'] = convert_to_initial_state('burner3', [-0.92, -0.01])
initial_states['light_switch'] = convert_to_initial_state(['light_switch'], [[-0.69, -0.05]])
initial_states['slide_cabinet'] = convert_to_initial_state(['slide_cabinet'], [[0.37]])
initial_states['hinge_cabinet'] = convert_to_initial_state(['hinge_cabinet'], [[0., 1.45]])

initial_states['micro_hinge'] = convert_to_initial_state(['microwave', 'hinge_cabinet'], [[-0.7], [0., 1.45]])
initial_states['micro_slide'] = convert_to_initial_state(['microwave', 'slide_cabinet'], [[-0.7], [0.37]])
initial_states['micro_light'] = convert_to_initial_state(['microwave', 'light_switch'], [[-0.7], [-0.69, -0.05]])
initial_states['light_slide'] = convert_to_initial_state(['light_switch', 'slide_cabinet'], [[-0.69, -0.05], [0.37]])
initial_states['light_hinge'] = convert_to_initial_state(['light_switch', 'hinge_cabinet'], [[-0.69, -0.05], [0., 1.45]])
initial_states['slide_hinge'] = convert_to_initial_state(['slide_cabinet', 'hinge_cabinet'], [[0.37], [0., 1.45]])

initial_states['all_pairs'] = np.array([initial_states['micro_hinge'].copy(),
                                        initial_states['micro_slide'].copy(),
                                        initial_states['micro_light'].copy(),
                                        initial_states['light_slide'].copy(),
                                        initial_states['light_hinge'].copy(),
                                        initial_states['slide_hinge'].copy()])
class Kitchen(KitchenTaskRelaxV1):

  def __init__(self, task="all_pairs", reward_type="dense"):
    self._initial_states = copy.deepcopy(initial_states)
    self._goal_states = copy.deepcopy(goal_states)
    if reward_type != 'dense':
        raise ValueError("Kitchen environment only supports dense rewards.")

    self._viewers = {}
    self.viewer = None
    self._reward_type = reward_type
    self._task = task
    super().__init__()

  def get_task(self):
    return self._task
  
  def get_init_states(self):
    return self._initial_states['all_pairs']

  def _get_obs(self):
    ob = super()._get_obs()
    return ob

  def get_next_goal(self):
    return self._goal_states[0]

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()
    self.set_goal(goal)

  def reset_model(self):
    reset_pos = self.init_qpos[:].copy()
    reset_vel = self.init_qvel[:].copy()

    if self._task == 'all_pairs':
      random_idx = np.random.randint(initial_states['all_pairs'].shape[0])
      reset_pos[9:] = self._initial_states[self._task][random_idx, 9:]
    else:
      reset_pos[9:] = self._initial_states[self._task][9:]

    self.robot.reset(self, reset_pos, reset_vel)
    for _ in range(10):
        new_pos = self.midpoint_pos
        self.sim.data.mocap_pos[:] = new_pos.copy()

    a = np.zeros(9)
    for _ in range(10):
        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

    self.reset_goal()
    return self._get_obs()
  
  def _get_reward_n_score(self, obs_dict):
    reward_dict = {}
    if isinstance(obs_dict, dict):
      obs = np.append(np.append(obs_dict['qp'], obs_dict['obj_qp']), obs_dict['goal'])
    else:
      obs = obs_dict
    
    task_to_site = {'microwave': 'microhandle_site',
                    'hinge_cabinet': 'hinge_site2',
                    'slide_cabinet': 'slide_site',
                    'burner0': 'knob1_site',
                    'burner1': 'knob2_site',
                    'burner2': 'knob3_site',
                    'burner3': 'knob4_site',
                    'light_switch': 'light_site',}

    reward_dict['true_reward'] = -10 * np.linalg.norm(obs[9:23] - obs[9+23:23+23])
    
    reaching_component = False
    for key in component_to_state_idx.keys():
      if key == 'arm':
        continue

      cur_idxs = np.array(component_to_state_idx[key])
      num_idxs = len(component_to_state_idx[key])
      if np.linalg.norm(obs[cur_idxs] - obs[cur_idxs + 23]) < num_idxs * 0.01:
        reward_dict['true_reward'] += 1
      elif not reaching_component:
        reaching_component = True
        reward_dict['true_reward'] += -0.5 * np.linalg.norm(self.sim.data.mocap_pos[0] - \
                                            self.sim.data.get_site_xpos(task_to_site[key]))
    reward_dict['r_total'] = reward_dict['true_reward']

    score = 0.
    return reward_dict, score

  def compute_reward(self, obs):
    return self._get_reward_n_score(obs)[0]['r_total']

  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()
    return bool(np.linalg.norm(obs[9:23] - obs[9+23:23+23]) <= 0.3)

  def step(self, a, b=None):
    obs, reward, done, info = super().step(a, b)
    return obs, reward, done, info

  # functions for rendering
  def viewer_setup(self):
    self.viewer.cam.distance = 3.0
    self.viewer.cam.elevation = -30
    self.viewer.cam.azimuth = 120

  def _get_viewer(self, mode):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if mode == 'human':
        self.viewer = mujoco_py.MjViewer(self.sim)
      if 'rgb_array' in mode:
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
      self._viewers[mode] = self.viewer

    self.viewer_setup()
    return self.viewer
  
  def render(self, mode='human', width=640, height=480):
    if mode == 'human':
      self._get_viewer(mode).render()
    elif mode == 'rgb_array':
      self._get_viewer(mode).render(width, height)
      return self.viewer.read_pixels(width, height, depth=False)[::-1, :, :]
    else:
      raise ValueError("mode can only be either 'human' or 'rgb_array'")

  def close(self):
    if self.viewer is not None:
      if isinstance(self.viewer, mujoco_py.MjViewer):
        glfw.destroy_window(self.viewer.window)
      self.viewer = None