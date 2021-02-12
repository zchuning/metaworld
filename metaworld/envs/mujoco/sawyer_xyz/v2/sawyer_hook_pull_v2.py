import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_stick_push_v2 import SawyerStickPushEnvV2

class SawyerHookPullEnvV2(SawyerStickPushEnvV2):
    def __init__(self):

        liftThresh = 0.04
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.58, 0.000)
        obj_high = (-0.03, 0.62, 0.001)
        goal_low = (0.399, 0.55, 0.0199)
        goal_high = (0.401, 0.6, 0.0201)

        super().__init__()

        self.init_config = {
            'stick_init_pos': np.array([-0.1, 0.6, 0.02]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config['stick_init_pos']
        self.stick_init_pos = self.init_config['stick_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        # For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.0])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_hook_obj.xml')
    
    @_assert_task_is_set
    def step(self, action):
        ob = SawyerXYZEnv.step(self, action)
        reward, _, reachDist, pickRew, _, pushDist, placeDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        obj_pos = self._get_site_pos('insertion') + np.array([.0, .09, .0]),
        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': pushDist,
            'success': float(np.linalg.norm(self._target_pos - obj_pos) <= 0.05)
        }
        # print(info['success'])
        # print(info['success'], self._target_pos, obj_pos, np.linalg.norm(self._target_pos - obj_pos))
        return ob, reward, False, info

    
    # def _get_pos_objects(self):
    #     return np.hstack((
    #         self.get_body_com('stick').copy(),
    #         self._get_site_pos('insertion') + np.array([.0, .09, .0]),
    #     ))
    #
    # def _set_stick_xyz(self, pos):
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     qpos[9:12] = pos.copy()
    #     qvel[9:15] = 0
    #     self.set_state(qpos, qvel)
    #
    # def _set_obj_xyz(self, pos):
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     qpos[16:18] = pos.copy()
    #     qvel[16:18] = 0
    #     self.set_state(qpos, qvel)
    #
    def reset_model(self):
        self._reset_hand()
        self.stick_init_pos = self.init_config['stick_init_pos']
        self._target_pos = np.array([-0.03, 0.5, 0.13])
        self.stickHeight = self.get_body_com('stick').copy()[2]
        self.heightTarget = self.stickHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.stick_init_pos = np.concatenate((goal_pos[:2], [self.stick_init_pos[-1]]))
            self._target_pos = np.concatenate((goal_pos[-3:-1], [self.stick_init_pos[-1]]))

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com('object').copy()
        self.maxPlaceDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self.stick_init_pos)) + self.heightTarget
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])

        return self._get_obs()

    def compute_reward(self, actions, obs):


        return [0, 0, 0, 0, 0, 0, 0]
