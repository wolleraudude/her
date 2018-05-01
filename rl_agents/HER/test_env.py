import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from os import path

register(
    id='TestEnv-v0',
    entry_point='test_env:TestEnv'
)

class TestEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed = 100.
        self.max_torque = 6.
        self.dt = .01

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()
        self.reset()
        self.sample_goal()

    def seed(self, seed=1234):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = u * self.max_torque
        u = np.clip(u, -self.max_torque, self.max_torque)[0] # not really necessary

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        obs = self._get_obs()
        r, d = self.reward(self.goal, obs)

        return obs, r, d, {}

    def reward(self, x_target, x):
        if abs(x_target[0] - x[0]) < 0.1 and \
        abs(x_target[1] - x[1]) < 0.1 and \
        abs(x_target[2] - x[2]) < 0.01:
            return 1, True
        else:
            return 0, False

    def sample_goal(self):
        high = np.array([np.pi, 1])
        theta, _ = self.np_random.uniform(low=-high, high=high)
        self.goal = np.array([np.cos(theta), np.sin(theta), 0])
        return self.goal

    def reset(self):
        high = np.array([np.pi, 0])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot / self.max_speed])

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)