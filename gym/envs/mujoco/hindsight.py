import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherHindEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    # def _step(self, a):
    #     vec = self.get_body_com("fingertip")-self.get_body_com("target")
    #     reward_dist = - np.linalg.norm(vec)
    #     reward_ctrl = - np.square(a).sum()
    #     reward = reward_dist + reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        reward = 0.0 if reward_dist < 0.03 else -1.0
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    # def hindsight_reward(self, state_pair, a, goal):
    #     # Notice that the last dimension is always reached.
    #     self.do_simulation(a, self.frame_skip)
    #     obs, new_obs = state_pair[0], state_pair[1]
    #     pos, new_pos = self.compute_position(obs), self.compute_position(new_obs)
    #     obs[4:6], new_obs[4:6] = goal, goal
    #     obs[-3:-1] = pos - goal
    #     new_obs[-3:-1] = new_pos - goal

    #     vec = new_pos - goal
    #     reward_dist = np.linalg.norm(vec)
    #     reward = 0.0 if reward_dist < 0.05 else -1.0
    #     return reward, obs, new_obs

    # def hindsight_reward(self, state_pair, a, goal):
    #     # Notice that the last dimension is always reached.
    #     obs, new_obs = state_pair[0], state_pair[1]
    #     pos, new_pos = self.compute_position(obs), self.compute_position(new_obs)
    #     obs[-3:-1] = pos - goal
    #     new_obs[-3:-1] = new_pos - goal

    #     vec = new_pos - goal
    #     reward_dist = np.linalg.norm(vec)
    #     reward = 0.0 if reward_dist < 0.01 else -1.0
    #     done = True if reward==0.0 else False
    #     return reward, obs, new_obs, done

    def hindsight_reward(self, state_pair, a, goal):
        # Notice that the last dimension is always reached.
        obs, new_obs = state_pair[0], state_pair[1]
        pos, new_pos = self.compute_position(obs), self.compute_position(new_obs)
        obs[-3:-1] = goal
        new_obs[-3:-1] = goal

        vec = new_pos - goal
        reward_dist = np.linalg.norm(vec)
        reward = 0.0 if reward_dist < 0.03 else -1.0
        done = True if reward==0.0 else False
        return reward, obs, new_obs, done

    # def compute_position(self, state):
    #     return state[-3:-1] + self.goal

    def compute_position(self, state):
        return state[4:6]

    # def sample_goal(self, pos):
    #     while True:
    #         random = self.np_random.uniform(low=-.04, high=.04, size=2)
    #         if np.linalg.norm(random) < 0.01:
    #             break
    #     return pos+random

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:2],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip"),
            self.get_body_com("target"), # goal concat
        ])

    # def _get_obs(self):
    #     theta = self.model.data.qpos.flat[:2]
    #     return np.concatenate([
    #         np.cos(theta),
    #         np.sin(theta),
    #         self.model.data.qvel.flat[:2],
    #         self.get_body_com("fingertip") - self.get_body_com("target"),
    #     ])

    # def _get_obs(self):
    #     theta = self.model.data.qpos.flat[:2]
    #     return np.concatenate([
    #         np.cos(theta),
    #         np.sin(theta),
    #         self.model.data.qpos.flat[2:],
    #         self.model.data.qvel.flat[:2],
    #         self.get_body_com("fingertip") - self.get_body_com("target"),
    #     ])