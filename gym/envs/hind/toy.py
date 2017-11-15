import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class ToyEnv(gym.Env):
    """Simple a bit-flipping environment environment.

    It is the implementation of the toy environment described in the 
    hindsight experience replay paper. https://arxiv.org/pdf/1707.01495.pdf.
    
    The state space is S = {0, 1}^n, the action space is {0,1,...,n-1} for 
    some integer n in which executing the i-th action flips the i-th bit of the state.
    For every episode we sample uniformly an initial state as well as a target state 
    and the policy gets a reward of −1 as long as it is not in the target state, 
    i.e. r_g(s, a) = −[s != g].

    """

    def __init__(self, space_size=5):
        self.action_space = spaces.Discrete(space_size)
        self.observation_space = spaces.Tuple(tuple([spaces.Discrete(2) for i in range(space_size)]))
        # self._seed()
        # self._reset()

    def set_space_size(self, space_size):
        self.action_space = spaces.Discrete(space_size)
        self.observation_space = spaces.Tuple(tuple([spaces.Discrete(2) for i in range(2*space_size)]))
        self.space_size = space_size

        self._seed()
        self._reset()        

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # assert self.observation_space.contains(goal), "%r (%s) invalid" % (goal, type(goal))

        self.steps += 1
        done = False if self.steps < self.space_size else True
        self.state[action] = 1 - self.state[action]
        coin_state = self.state[:self.space_size]
        reward = 0. if (coin_state == self.goal).all() else -1.0
        if reward == 0.: done=True
        return np.array(self.state), reward, done, {}

    def hindsight_reward(self, state, action, goal):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # assert self.observation_space.contains(goal), "%r (%s) invalid" % (goal, type(goal))

        state[action] = 1 - state[action]
        coin_state = state[:self.space_size]
        reward = 0. if (coin_state == goal).all() else -1.0
        done = True if reward == 0. else False
        return np.array(state), reward, done

    def _reset(self):
        self.coin_state = np.array(np.random.randint(2, size=self.space_size), dtype=np.float32)
        self.goal = self._sample_goal()
        self.state = np.concatenate((self.coin_state, self.goal))
        self.steps = 0
        return np.array(self.state)

    def _sample_goal(self):
        goal = np.random.randint(2, size=self.space_size)
        goal = np.array(goal, dtype=np.float32)
        return goal