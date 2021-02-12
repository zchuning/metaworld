import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move

add_stick = np.array([-.12, .0, .03])

class SawyerHookPullV2Policy(Policy):
    
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'stick_pos': obs[3:6],
            'obj_pos': obs[6:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })
        
        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_xyz(o_d), p=10.)
        action['grab_pow'] = self._grab_pow(o_d)

        # print(action.array)
        return action.array
    
    def _desired_xyz(self, o_d):
        hand_pos = o_d['hand_pos']
        stick_pos = o_d['stick_pos'] + add_stick
        # stick_obj_pos = o_d['stick_pos'] + np.array([.12, .0, .0])
        thermos_pos = o_d['obj_pos']
        thermos_goal_pos = np.array([-0.03, 0.5, 0.13])
        goal_pos = thermos_goal_pos + np.array([-.35, .1, .0])
        # goal_pos = thermos_pos + np.array([-.35, .1, .0])
        wp1_pos = thermos_pos + np.array([-.15, 0.17, .0])

        if np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02 or abs(hand_pos[2] - stick_pos[2]) > 0.1:
            # print("HAHA")
            self.wp1_completed = False
          
        # print(self.wp1_completed)
        
        if abs(stick_pos[0] - wp1_pos[0]) > 0.04 and not self.wp1_completed:
            # print('wp')
            if np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02:
                return stick_pos + np.array([0., 0., 0.1])
            elif abs(hand_pos[2] - stick_pos[2]) > 0.02:
                return stick_pos
            elif abs(stick_pos[1] - wp1_pos[1]) > 0.02:
                return np.array([stick_pos[0], wp1_pos[1], stick_pos[2]])
            elif abs(stick_pos[2] - wp1_pos[2]) > 0.02:
                return np.array([stick_pos[0], *wp1_pos[1:]])
            else:
                return wp1_pos
        else:
            self.wp1_completed = True
            if abs(stick_pos[1] - goal_pos[1]) > 0.02:
                return np.array([stick_pos[0], goal_pos[1], stick_pos[2]])
            return goal_pos

    @staticmethod
    def _grab_pow(o_d):
        hand_pos = o_d['hand_pos']
        stick_pos = o_d['stick_pos'] + add_stick

        if np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02 or abs(hand_pos[2] - stick_pos[2]) > 0.05:
            return -1.0
        else:
            return +1.0
