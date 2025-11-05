# MIT License

# Copyright (c) 2019 CleanRL developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import numpy as np
from gymnasium import spaces


FRAME_SIZE = {
    "BreakoutNoFrameskip-v4": "[32:, 8:-8]",
    "PongNoFrameskip-v4": "[34:-16]",
    # Learned the cropping from https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html
    "CarRacing-v2": "[:84, 6:90]",
    "MsPacmanNoFrameskip-v4": "[:-39]",
}


ACTION_MAPPER = {
    "CarRacing-v2": {0: "do nothing", 1: "steer right",
                     2: "steer left", 3: "gas", 4: "break"},
    "PongNoFrameskip-v4": {0: "do nothing", 1: "do nothing (fire)", 2: "go up",
                           3: "go down", 4: "go up (fire)", 5: "go down (fire)"},
    "BreakoutNoFrameskip-v4": {0: "noop", 1: "fire", 2: "right", 3: "left"},
    "MsPacmanNoFrameskip-v4": {0: "noop", 1: "up", 2: "right", 3: "left",
                               4: "down", 5: "upright", 6: "upleft",
                               7: "downright", 8: "downleft"},
}


class CropObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, env_id: str) -> None:
        if env_id == "BreakoutNoFrameskip-v4":
            shape = (178, 144, 3)
        elif env_id == "PongNoFrameskip-v4":
            shape = (160, 160, 3)
        elif env_id == "CarRacing-v2":
            shape = (84, 84, 3)
        elif env_id == "MsPacmanNoFrameskip-v4":
            shape = (171, 160, 3)
        else:
            raise NotImplementedError

        gym.ObservationWrapper.__init__(self, env)

        self.env_id = env_id
        self.observation_space = spaces.Box(low=0, high=255, shape=shape,
                                            dtype=np.uint8)

    def observation(self, observation):
        observation = eval(f"observation{FRAME_SIZE[self.env_id]}")
        return observation.reshape(self.observation_space.shape)


def make_env(env_id, seed, idx, capture_video, folder):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, folder)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = CropObservation(env, env_id)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.get_wrapper_attr('seed')(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    def thunk_carracing():
        # 1: steer right, 2: steer left, 3: gas
        env = gym.make(env_id, continuous=False, render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, folder)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CropObservation(env, env_id)
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    match env_id:
        case "CarRacing-v2":
            return thunk_carracing
        case _:
            return thunk
