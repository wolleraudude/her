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
        self.max_speed = 8.
        self.max_torque = 6.
        self.dt = .05
        self.viewer = None
        self.goal_display = None

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
        self.last_u = u

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        obs = self._get_obs()
        r, d = self.reward(self.goal, obs)

        return obs, r, d, {}

    def reward(self, x_target, x):
        delta_th = np.arccos(x_target[0]) - np.arccos(x[0])
        delta_thdot = x_target[2] - x[2]
        cost = angle_normalize(delta_th)**2 + 0.1 * delta_thdot**2
        cost /= np.pi**2 + 0.5 * self.max_speed**2

        if abs(delta_th) < 0.1 * np.pi and abs(delta_thdot) < 0.01:
            return -cost, True

        else:
            return -cost, False


    def sample_goal(self):
        rand_val = self.np_random.randint(-5, 5)
        rand_val /= 5.
        theta = rand_val * np.pi
        self.goal = np.array([np.cos(theta), np.sin(theta), 0])
        return self.goal

    def reset(self):
        high = np.array([np.pi, 0])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot / self.max_speed])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2., np.abs(self.last_u) / 2. )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_goal(self, mode='human'):

        if self.goal_display is None:
            from gym.envs.classic_control import rendering
            self.goal_display = rendering.Viewer(500, 500)
            self.goal_display.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.goal_display.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.goal_display.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.goal_display.add_onetime(self.img)
        self.pole_transform.set_rotation(self.goal[0] + np.pi / 2)

        return self.goal_display.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
        self.viewer = None

    def close_goal(self):
        if self.goal_display: self.goal_display.close()
        self.goal_display = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)