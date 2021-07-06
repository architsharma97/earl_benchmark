"A simplified manipulation environment for objects."

import os

from gym import spaces
from gym.envs.mujoco import MujocoEnv

import random
import numpy as np

initial_states = np.array([[0.0, 0.0, 2.5, 0.0, -1., -1.]])
goal_states = np.array([[0.0, 0.0, -2.5, -1.0, -1., -1.],
                        [0.0, 0.0, -2.5,  1.0, -1., -1.],
                        [0.0, 0.0,  0.0,  2.0, -1., -1.],
                        [0.0, 0.0,  0.0, -2.0, -1., -1.],
                       ])

class TabletopManipulation(MujocoEnv):

  FILE_TREE = "tabletop_manipulation.xml"
  MODEL_PATH_TREE = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "tabletop_assets", FILE_TREE)

  def __init__(self,
               task_list="rc_o",
               reward_type="dense",
               reset_at_goal=False):
    # r->red, b->blue y->yellow, k->black, o->orange, p->purple
    self.object_colors = [
        "r",
    ]
    # c-> cube, s->sphere, r->cylinder
    self.objects = [
        "c",
    ]
    self.target_colors = ["o", "k", "p", "b"]

    # TODO: Fill these in automatically
    # Dict of object to index in qpos
    self.object_dict = {
        (0, 0): [2, 3],
    }
    # Dict of targets to index in geom_pos
    self.target_dict = {0: 8, 1: 6, 2: 7, 3: 5}

    self.attached_object = (-1, -1)
    self.threshold = 0.4
    self.move_distance = 0.2

    self._task_list = task_list
    self._reward_type = reward_type
    self.initial_state = initial_states.copy()[0]
    self.goal = self.initial_state.copy()
    self._reset_at_goal = reset_at_goal  # use only in train envs without resets
    super().__init__(model_path=self.MODEL_PATH_TREE, frame_skip=15)

  def _get_obs(self):
    return np.concatenate([
        self.sim.data.qpos.flat[:4],  # remove the random joint
        np.asarray(self.attached_object),
        self.goal,
    ]).astype("float32")

  def get_next_goal(self):
    # the gripper should return to the original position (def in sparse reward)
    goal = self.initial_state.copy()

    cur_task = random.sample(self._task_list.split("-"), 1)[0]
    for task in cur_task.split("__"):
      color_to_move = self.object_colors.index(task.split("_")[0][0])
      object_to_move = self.objects.index(task.split("_")[0][1])
      target_index = self.target_colors.index(task.split("_")[1])

      obj_idx = self.object_dict[(color_to_move, object_to_move)]
      target_pos = self.model.geom_pos[self.target_dict[target_index]][:2]
      goal[obj_idx[0]:obj_idx[1] + 1] = target_pos  # the object

    return goal

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()
    self.goal = goal

  def reset(self):
    self.attached_object = (-1, -1)
    full_qpos = np.zeros((5,))

    if self._reset_at_goal:
      self.reset_goal()
      full_qpos[:4] = self.goal[:4]
      full_qpos[:4] = np.random.uniform(-2.5, 2.5, size=(4,))
      full_qpos[4] = -10
      curr_qvel = self.sim.data.qvel.copy()
    else:
      full_qpos[:4] = self.initial_state[:4]
      # the joint is required for the gripping actuator
      full_qpos[4] = -10

      curr_qvel = self.sim.data.qvel.copy()
      self.reset_goal()

    self.set_state(full_qpos, curr_qvel)
    self.sim.forward()
    return self._get_obs()

  def step(self, action):
    # rescale and clip action
    action = np.clip(action, np.array([-1.] * 3), np.array([1.] * 3))
    lb, ub = np.array([-0.2, -0.2, -0.2]), np.array([0.2, 0.2, 0.2])
    action = lb + (action + 1.) * 0.5 * (ub - lb)

    self.move(action)
    next_obs = self._get_obs()
    reward = self.compute_reward(next_obs)
    done = False
    return next_obs, reward, done, {}

  def move(self, action):
    current_fist_pos = self.sim.data.qpos[0:2].flatten()
    curr_action = action[:2]

    if action[-1] > 0:
      if self.attached_object == (-1, -1):
        self._dist_of_cur_held_obj = np.inf  # to ensure the closest object is grasped when multiple objects are within threshold
        for k, v in self.object_dict.items():
          curr_obj_pos = np.array([self.sim.data.qpos[i] for i in v])
          dist = np.linalg.norm((current_fist_pos - curr_obj_pos))
          if dist < self.threshold and dist < self._dist_of_cur_held_obj:
            self.attached_object = k
            self._dist_of_cur_held_obj = dist
    else:
      self.attached_object = (-1, -1)

    next_fist_pos = current_fist_pos + curr_action
    next_fist_pos = np.clip(next_fist_pos, -2.8, 2.8)
    if self.attached_object != (-1, -1):
      current_obj_pos = np.array([
          self.sim.data.qpos[i] for i in self.object_dict[self.attached_object]
      ])
      current_obj_pos += (next_fist_pos - current_fist_pos)
      current_obj_pos = np.clip(current_obj_pos, -2.8, 2.8)

    # Setting the final positions
    curr_qpos = self.sim.data.qpos.copy()
    curr_qpos[0] = next_fist_pos[0]
    curr_qpos[1] = next_fist_pos[1]
    if self.attached_object != (-1, -1):
      for enum_n, i in enumerate(self.object_dict[self.attached_object]):
        curr_qpos[i] = current_obj_pos[enum_n]

    # dummy joint
    curr_qpos[4] = -10
    curr_qvel = self.sim.data.qvel.copy()
    self.set_state(curr_qpos, curr_qvel)
    self.sim.forward()

  def compute_reward(self, obs):
    if self._reward_type == "sparse":
      reward = float(self.is_successful(obs=obs))
    elif self._reward_type == "dense":
      # remove gripper, attached object from reward computation
      reward = -np.linalg.norm(obs[2:4] - obs[8:-2])
      for obj_idx in range(1, 2):
        reward += 2. * np.exp(
            -(np.linalg.norm(obs[2 * obj_idx:2 * obj_idx + 2] -
                             obs[2 * obj_idx + 6:2 * obj_idx + 8])**2) / 0.01)
      # gripper reward
      if self._task_list == "rc_o":
        grip_to_object = np.linalg.norm(obs[:2] - obs[2:4])
        reward += -grip_to_object
        reward += 0.5 * np.exp(-(grip_to_object**2) / 0.01)

    return reward

  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()

    return np.linalg.norm(obs[:4] - obs[6:-2]) <= 0.2
