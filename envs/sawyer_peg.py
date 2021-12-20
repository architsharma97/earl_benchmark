"""Sawyer environment to pickup and insert pegs.

The observation is configured to included the gripper position, gripper distance and head of the object.
The sparse reward function only considers the position of the object, not the position of the gripper.
"""
import os

import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from scipy.spatial.transform import Rotation

import mujoco_py
import numpy as np

initial_states = np.array(
      [[ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.00313463,
         0.68326396,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        , -0.04035005,
         0.67949003,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.02531051,
         0.6074387 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.05957219,
         0.6271171 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        , -0.07566337,
         0.62575287,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        , -0.01177235,
         0.55206996,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.02779735,
         0.54707706,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.01835314,
         0.5329686 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.02690855,
         0.6263067 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.01766127,
         0.59630984,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.0560186 ,
         0.6634998 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        , -0.03950658,
         0.6323736 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        , -0.03216827,
         0.5247563 ,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.01265727,
         0.69466716,  0.02      ],
       [ 0.00615235,  0.6001898 ,  0.19430117,  1.        ,  0.05076993,
         0.6025737 ,  0.02      ]])

goal_states = np.array([[0.0, 0.6, 0.2, 1.0, -0.3 + 0.03, 0.6, 0.0 + 0.13]])

class SawyerPegV2(SawyerXYZEnv):
  max_path_length = int(1e8)
  TARGET_RADIUS = 0.05

  def __init__(self, reward_type='dense', reset_at_goal=False):

    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (.0, 0.5, 0.02)
    obj_high = (.2, 0.7, 0.02)
    goal_low = (-0.35, 0.4, -0.001)
    goal_high = (-0.25, 0.7, 0.001)

    super().__init__(
        self.model_name,
        hand_low=hand_low,
        hand_high=hand_high,
    )

    self.init_config = {
        'obj_init_pos': np.array([0, 0.6, 0.02]),
        'hand_init_pos': np.array([0, 0.6, 0.2]),
    }

    self.initial_states = initial_states
    self.goal_states = goal_states

    self.obj_init_pos = self.init_config['obj_init_pos']
    self.hand_init_pos = self.init_config['hand_init_pos']
    self.peg_head_pos_init = self._get_site_pos('pegHead')
    self.goal = goal_states[0]
    self._target_pos = self.goal[4:]


    self._partially_observable = False
    self._set_task_called = True
    self._freeze_rand_vec = False

    self._reset_at_goal = reset_at_goal
    self._reward_type = reward_type
    
    self._random_reset_space = Box(
      np.hstack((obj_low, goal_low)),
      np.hstack((obj_high, goal_high)),
    )
    self.goal_space = Box(
      np.array(goal_low) + np.array([.03, .0, .13]),
      np.array(goal_high) + np.array([.03, .0, .13])
    )

    self.metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': int(np.round(1.0 / self.dt))
    }

  @property
  def model_name(self):
    return os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "metaworld_assets/sawyer_xyz", 'sawyer_peg_insertion_side.xml')
  
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
    obj_pos = self._get_pos_objects()
    obs = np.concatenate([
        endeff_config, obj_pos, self.goal,
    ])
    return obs

  def get_next_goal(self):
    num_goals = self.goal_states.shape[0]
    goal_idx = np.random.randint(0, num_goals)
    return self.goal_states[goal_idx]

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()

    self.goal = goal
    self._target_gripper_distance = goal[3]
    self._target_pos = goal[4:]
    self._end_effector_goal = goal[:3]

    self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos

  @_assert_task_is_set
  def evaluate_state(self, obs, action):
    obj = obs[4:7]

    reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward, collision_box_front, ip_orig = self.compute_reward(obs, action)
    grasp_success = float(tcp_to_obj < 0.02 and (tcp_open > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]))
    success = float(obj_to_target <= self.TARGET_RADIUS)
    near_object = float(tcp_to_obj <= 0.03)

    info = {
        'success': success,
        'near_object': near_object,
        'grasp_success': grasp_success,
        'grasp_reward': grasp_reward,
        'in_place_reward': in_place_reward,
        'obj_to_target': obj_to_target,
        'unscaled_reward': reward,
    }

    return reward, info

  def _get_pos_objects(self):
    return self._get_site_pos('pegHead')

  def _get_quat_objects(self):
    return Rotation.from_matrix(self.data.get_site_xmat('pegHead')).as_quat()

  def reset_model(self):
    if not self._reset_at_goal:
      self._reset_hand()
      self.reset_goal()
      pos_box = self.goal_states[0][4:] - np.array([0.03, 0.0, 0.13])
      self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box
      
      pos_peg = self.obj_init_pos
      if self.random_init:
        pos_peg, _ = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
            pos_peg, _ = np.split(self._get_state_rand_vec(), 2)

      self.obj_init_pos = pos_peg
      self.peg_head_pos_init = self._get_site_pos('pegHead')
      self._set_obj_xyz(self.obj_init_pos)
    else:
      self._reset_hand()
      self.reset_goal()
      pos_box = self.goal_states[0][4:] - np.array([0.03, 0.0, 0.13])
      self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box

      goal_pos = self.goal_states[0][4:] - np.array([-0.1, 0., 0.])
      self.obj_init_pos = goal_pos + np.random.uniform(-0.02, 0.02, size=3)
      self._set_obj_xyz(self.obj_init_pos)
      self.peg_head_pos_init = self._get_site_pos('pegHead')

    return self._get_obs()

  def compute_reward(self, obs, action=None):
    tcp = obs[:3]
    obj = obs[4:7] - self._get_site_pos('pegHead') \
                   + self._get_site_pos('pegGrasp')
    obj_head = obs[4:7] 
    tcp_opened = obs[3]
    target = obs[11:14]

    tcp_to_obj = np.linalg.norm(obj - tcp)
    
    scale = np.array([1., 2., 2.])
    #  force agent to pick up object then insert
    obj_to_target = np.linalg.norm((obj_head - target) * scale)

    in_place_margin = np.linalg.norm((self.peg_head_pos_init - target) * scale)
    in_place = reward_utils.tolerance(obj_to_target,
                                bounds=(0, self.TARGET_RADIUS),
                                margin=in_place_margin,
                                sigmoid='long_tail',)
    ip_orig = in_place
    brc_col_box_1 = self._get_site_pos('bottom_right_corner_collision_box_1')
    tlc_col_box_1 = self._get_site_pos('top_left_corner_collision_box_1')

    brc_col_box_2 = self._get_site_pos('bottom_right_corner_collision_box_2')
    tlc_col_box_2 = self._get_site_pos('top_left_corner_collision_box_2')
    collision_box_bottom_1 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                               one=tlc_col_box_1,
                                                               zero=brc_col_box_1)
    collision_box_bottom_2 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                               one=tlc_col_box_2,
                                                               zero=brc_col_box_2)
    collision_boxes = reward_utils.hamacher_product(collision_box_bottom_2,
                                                    collision_box_bottom_1)
    in_place = reward_utils.hamacher_product(in_place,
                                             collision_boxes)

    if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
      object_grasped = 1.

    # this is the only part that requires the action, only with dense reward
    elif self._reward_type == 'dense':
      pad_success_margin = 0.03
      object_reach_radius=0.01
      x_z_margin = 0.005
      obj_radius = 0.0075
      object_grasped = self._gripper_caging_reward(action,
                                                   obj,
                                                   object_reach_radius=object_reach_radius,
                                                   obj_radius=obj_radius,
                                                   pad_success_thresh=pad_success_margin,
                                                   xz_thresh=x_z_margin,
                                                   high_density=True)
    elif self._reward_type == 'sparse':
      object_grasped = 0.

    in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                in_place)
    reward = in_place_and_object_grasped

    if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
      reward += 1. + 5 * in_place

    if obj_to_target <= self.TARGET_RADIUS:
      reward = 10.

    if self._reward_type == 'sparse':
      reward = float(self.is_successful(obs=obs))

    return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place, collision_boxes, ip_orig]

  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()

    return np.linalg.norm(obs[4:7] - obs[11:14]) <= self.TARGET_RADIUS

  def viewer_setup(self):
    self.viewer.cam.distance = 2.0
    self.viewer.cam.elevation = -20
    self.viewer.cam.azimuth = -150

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
