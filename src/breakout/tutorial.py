import gym
from baselines.common.atari_wrappers import wrap_deepmind
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter




#################################
#####      Environment     ######
#################################

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        frames = np.array(list(self.frames)).reshape(self.k, 84, 84) # 4,84,84
        frames = frames.transpose(1, 2, 0) # 84, 84, 4
        #print("frames", frames.shape)
        return frames

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def wrap_breakout_env(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

env = gym.make('Breakout-v0')
# env = make_atari('Breakout-v0')
env = wrap_deepmind(env)  
env = WrapPyTorch(env) # output (1, 84, 84)


observation = env.reset()

# ovservationの可視化
print(observation.reshape(84, 84).shape)
pilOUT = Image.fromarray(np.uint8(observation.reshape(84, 84)))
pilOUT.show()



while True:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())

    if done:
        break
        observation = env.reset()
        print(observation.shape)

# データを作る
np.random.seed(123)
x = np.random.randn(100)
y = x.cumsum()  # xの累積和
writer = SummaryWriter(log_dir="./logs")
for i in range(100):
    writer.add_scalar("x", x[i], i)
    writer.add_scalar("y", y[i], i)
writer.close()

env.close()