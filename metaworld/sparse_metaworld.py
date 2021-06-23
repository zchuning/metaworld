import numpy as np
import os
import threading
import gym


if 'MUJOCO_RENDERER' in os.environ:
  RENDERER = os.environ['MUJOCO_RENDERER']
else:
  RENDERER = 'glfw'

def get_device_id():
  return int(os.environ.get('GL_DEVICE_ID', 0))

def set_camera(cam, azimuth=None, elevation=None, distance=None, lookat=None):
  """ Sets camera parameters """
  if azimuth:
    cam.azimuth = azimuth
  if elevation:
    cam.elevation = elevation
  if distance:
    cam.distance = distance
  if lookat:
    for i in range(3):
      cam.lookat[i] = lookat[i]


class DreamerEnv():

  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    return self._get_obs(state)

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self, state):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)

    obs = {'image': image}
    if isinstance(state, dict):
      obs.update(state)
    else:
      obs['state'] = state
    return obs


class SparseMetaWorld(DreamerEnv):

  def __init__(self, task, action_repeat):
    super().__init__(action_repeat)
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz.v2 as sawyerv2
    assert 'V2' in task
    self.task_type = self.get_task_type(task)

    with self.LOCK:
        self._env = getattr(sawyerv2, task)()
    self._env.random_init = False
    self._env.max_path_length = np.inf

    self._action_repeat = action_repeat
    self._rand_goal = False
    self._rand_hand = self.task_type in ['reach', 'button', 'window', 'drawer']
    self._rand_obj = self.task_type in ['window', 'drawer', 'push', 'hammer']
    self._width = 64
    self._size = (self._width, self._width)
    self.rendered_goal = False

    self._offscreen = MjRenderContext(self._env.sim, True, get_device_id(), RENDERER, True)
    if "SawyerHammerEnv" in task:
      set_camera(self._offscreen.cam, azimuth=220, elevation=-140, distance=0.8, lookat=[0.2, 0.65, -0.1])
    else:
      set_camera(self._offscreen.cam, azimuth=205, elevation=-165, distance=2.6, lookat=[1.1, 1.1, -0.1])

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      reward = info['success'] # Use sparse reward
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info
  
  def get_task_type(self, task):
    if 'SawyerReachEnv' in task:
      return 'reach'
    if 'SawyerButtonPressEnv' in task:
      return 'button'
    if 'SawyerWindowCloseEnv' in task or 'SawyerWindowOpenEnv' in task:
      return 'window'
    if 'SawyerDrawerCloseEnv' in task or 'SawyerDrawerOpenEnv' in task:
      return 'drawer'
    if 'SawyerPushEnv' in task:
      return 'push'
    if 'SawyerStickPushEnv' in task:
      return 'thermos'
    if 'SawyerHammerEnv' in task:
      return 'hammer'

  def reset(self):
    self.rendered_goal = False

    if self.task_type == 'reach' or self.task_type == 'button':
      self._env.hand_init_pos = np.random.uniform(
        self._env.hand_low,
        self._env.hand_high,
        size=(self._env.hand_low.size)
      )
      
    elif self.task_type == 'window':
      slider_init_pos = np.random.uniform(0, 0.2)
      self._env.init_config['slider_init_pos'] = slider_init_pos
      self._env.hand_init_pos = self._env.obj_init_pos + np.array([slider_init_pos, -0.1, 0.05])
      self._env.hand_init_pos += np.random.uniform((-0.15, -0.3, -0.15), (0.15, 0, 0.15), size=(3))
      
    elif self.task_type == 'drawer':
      drawer_init_pos = np.random.uniform(0, 0.15)
      self._env.init_config['drawer_init_pos'] = drawer_init_pos
      # -0.16 is the protrusion of the handle
      self._env.hand_init_pos = self._env.obj_init_pos + np.array([0, -0.16-drawer_init_pos, 0.15])
      self._env.hand_init_pos += np.random.uniform((-0.15, -0.15, 0), (0.15, 0.15, 0.3), size=(3))

    if self.task_type == 'push':
      obj_init_pos = np.random.uniform(
        (-0.2, 0.6, 0.02),
        (0.4, 0.9, 0.02),
        size=(3)
      )
      self._env.init_config['obj_init_pos'] = obj_init_pos
      # Initialize hand above object
      self._env.hand_init_pos = obj_init_pos
      # Initialize hand with fixed height, can be overridden by rand_hand
      self._env.hand_init_pos[2] = 0.05
      
    elif self.task_type == 'hammer':
      hammer_init_pos = np.random.uniform(
        (-0.1, 0.3, 0.04),
        (0.3, 0.7, 0.04),
        size=(3)
      )
      self._env.init_config['hammer_init_pos'] = hammer_init_pos
      # Initialize hand with fixed displacement, can be overridden by rand_hand
      self._env.hand_init_pos = hammer_init_pos + np.array([0., -0.1, 0.16])
    return super().reset()