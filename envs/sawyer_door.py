"""Sawyer environment for opening and closing a door."""

import os
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_close_v2 import SawyerDoorCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set

import mujoco_py
import numpy as np

initial_states = np.array([[0.00591636, 0.39968333, 0.19493164, 1.0,
                           0.01007495, 0.47104556, 0.10003595]])
goal_states = np.array([[0.29072163, 0.74286009, 0.10003595, 1.0,
                        0.29072163, 0.74286009, 0.10003595]])

class SawyerDoorV2(SawyerDoorCloseEnvV2):
  max_path_length = int(1e8)

  def __init__(self, reward_type='dense', reset_at_goal=False):
    self._reset_at_goal = reset_at_goal

    super().__init__()
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (-0.25, 0.45, 0.1)
    obj_high = (0.35, 0.85, 0.15)
    goal_low = (-0.25, 0.45, 0.0999)
    goal_high = (0.35, 0.85, 0.1001)

    self.init_config = {
        'obj_init_angle': -np.pi / 3 if not self._reset_at_goal \
                          else 0,  # default initial angle
        # 'obj_init_angle': 0,  # reset initial angle
        'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
        'hand_init_pos': np.array(
            [0, 0.4, 0.2] if not self._reset_at_goal \
            else [0.29, 0.74, 0.1],
            dtype=np.float32),
    }

    # (end effector pos, handle pos)
    # -pi/3 initial state -> [0.00591636, 0.39968333, 0.19493164, 0.01007495, 0.47104556, 0.10003595]
    # 0 initial state -> [0.00591636, 0.39968333, 0.19493164, 0.29072163, 0.74286009, 0.10003595]

    self.goal = np.array([0.29072163, 0.74286009, 0.10003595, 1.0,
                          0.29072163, 0.74286009, 0.10003595])  # 0 angle state, goal for forward policy
    # self.goal = np.array([0.00591636, 0.39968333, 0.19493164, 1.0,
    #                       0.01007495, 0.47104556, 0.10003595]) # -pi/3 angle state, goal for reverse policy

    self.goal_states = goal_states.copy()
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.hand_init_pos = self.init_config['hand_init_pos']

    self.goal_space = Box(np.array(goal_low), np.array(goal_high))
    self._partially_observable = False
    self._set_task_called = True
    self._target_pos = self.goal[3:]
    self._reward_type = reward_type

    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': int(np.round(1.0 / self.dt))
    }

  @property
  def model_name(self):
    return os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "metaworld_assets/sawyer_xyz", 'sawyer_door_pull.xml')

  @property
  def observation_space(self):
    obj_low = np.full(3, -np.inf)
    obj_high = np.full(3, +np.inf)
    goal_low = self.goal_space.low
    goal_high = self.goal_space.high
    gripper_low = -1
    gripper_high = 1
    return Box(
      np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, self._HAND_SPACE.low, gripper_low, goal_low)),
      np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, self._HAND_SPACE.high, gripper_high, goal_high))
    )

  def _get_obs(self):
    obs = super()._get_obs()
    # xyz and gripper distance for end effector
    endeff_config = obs[:4]
    obj_pos = obj_pos = self._get_pos_objects()
    obs = np.concatenate([
        endeff_config, obj_pos, self.goal,
    ])
    return obs

  # need to expose the default goal, useful for multi-goal settings
  def get_next_goal(self):
    return self.goal_states[0]

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()

    self.goal = goal
    self._target_gripper_distance = goal[3]
    self._target_pos = self._handle_goal = goal[4:]
    self._end_effector_goal = goal[:3]

    self.sim.model.site_pos[self.model.site_name2id('goal')] = self._handle_goal

  def reset_model(self):
    self._reset_hand()
    self.objHeight = self.data.get_geom_xpos('handle')[2]
    if self.random_init:
      # add noise to the initial position of the door
      initial_position = self.obj_init_angle
      initial_position += np.random.uniform(0, np.pi / 20) if not self._reset_at_goal \
        else np.random.uniform(-np.pi / 20, 0)
      self.sim.model.body_pos[self.model.body_name2id(
          'door')] = self.obj_init_pos
      self._set_obj_xyz(initial_position)

    self.reset_goal()

    return self._get_obs()
  
  @_assert_task_is_set
  def evaluate_state(self, obs, action):
      reward, obj_to_target, in_place = self.compute_reward(obs, action)
      info = {
          'obj_to_target': obj_to_target,
          'in_place_reward': in_place,
          'success': float(obj_to_target <= 0.08),
          'near_object': 0.,
          'grasp_success': 1.,
          'grasp_reward': 1.,
          'unscaled_reward': reward,
      }
      return reward, info

  def compute_reward(self, obs, actions=None):
    _TARGET_RADIUS = 0.05
    tcp = obs[:3]
    obj = obs[4:7]
    target = obs[11:14]

    tcp_to_target = np.linalg.norm(tcp - target)
    tcp_to_obj = np.linalg.norm(tcp - obj)
    obj_to_target = np.linalg.norm(obj - target)

    in_place_margin = np.linalg.norm(self.obj_init_pos - target)
    in_place = reward_utils.tolerance(obj_to_target,
                                bounds=(0, _TARGET_RADIUS),
                                margin=in_place_margin,
                                sigmoid='gaussian',)

    hand_margin = np.linalg.norm(self.hand_init_pos - obj) + 0.1
    hand_in_place = reward_utils.tolerance(tcp_to_obj,
                                bounds=(0, 0.25*_TARGET_RADIUS),
                                margin=hand_margin,
                                sigmoid='gaussian',)

    reward = 3 * hand_in_place + 6 * in_place

    if obj_to_target < _TARGET_RADIUS:
        reward = 10

    if self._reward_type == 'sparse':
      reward = float(self.is_successful(obs=obs))
    
    return [reward, obj_to_target, hand_in_place]

  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()

    return np.linalg.norm(obs[4:7] - obs[11:14]) <= 0.02

  # functions for rendering
  def viewer_setup(self):
    self.viewer.cam.distance = 2.0
    self.viewer.cam.elevation = -20
    self.viewer.cam.azimuth = 20

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
  
  def render(self, mode='human'):
    if mode == 'human':
      self._get_viewer(mode).render()
    elif mode == 'rgb_array':
      return self.sim.render(
          640, 480,
          mode='offscreen',
          camera_name='topview'
      )[::-1, :, :]
    else:
      raise ValueError("mode can only be either 'human' or 'rgb_array'")

  def close(self):
    if self.viewer is not None:
      if isinstance(self.viewer, mujoco_py.MjViewer):
        glfw.destroy_window(self.viewer.window)
      self.viewer = None