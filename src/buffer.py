# The MIT License

# Copyright (c) 2019 Antonin Raffin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import gymnasium as gym
import torch
from torch.utils.data import Dataset
from typing import Optional
from tqdm import trange
from copy import deepcopy
from lightning import fabric
import joblib
from src.env import make_env


class RolloutDataset(Dataset):
    def __init__(self, envs, obs, output, value) -> None:
        action_space = envs.single_action_space
        self.obs = obs.clone().reshape((-1,) + envs.single_observation_space.shape)
        self.value = value.clone().reshape(-1)

        if isinstance(action_space, gym.spaces.Discrete):
            self.output = output.clone().reshape(
                (-1,) + (action_space.n,))
        elif isinstance(action_space, gym.spaces.Box):
            self.output = output.clone().reshape(
                (-1,) + action_space.shape)
        else:
            raise NotImplementedError(
                f"Action space: {action_space} is not implemented")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx: int):
        return (self.obs[idx], self.output[idx], self.value[idx])


class RolloutBuffer:
    def __init__(self, envs: gym.vector.SyncVectorEnv, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        action_space = envs.single_action_space
        observation_space = envs.single_observation_space

        self.n_envs = len(envs.envs)

        assert self.buffer_size % self.n_envs == 0, \
            "Please choose buffer size that is dividable by " \
            f"number of environments ({self.n_envs})."
        dim = (int(self.buffer_size / self.n_envs), self.n_envs)
        self.obs = torch.zeros(
            dim + observation_space.shape, dtype=torch.float32)
        self.value = torch.zeros(dim, dtype=torch.float32)

        if isinstance(action_space, gym.spaces.Discrete):
            output_dim = dim + (action_space.n,)
        elif isinstance(action_space, gym.spaces.Box):
            output_dim = dim + action_space.shape
        else:
            raise NotImplementedError(
                f"Action space: {action_space} is not implemented")

        self.output = torch.zeros(output_dim, dtype=torch.float32)

        self.pos = 0
        self.full = False
        self.updated = False

    def add(self, obs: torch.Tensor, output: torch.Tensor,
            value: Optional[torch.Tensor]) -> None:
        self.obs[self.pos] = obs.clone()
        self.output[self.pos] = output.clone()
        if value is not None:
            self.value[self.pos] = value.clone()

        self.pos += 1
        if self.buffer_size == self.pos:
            self.full = True
            self.pos = 0

    def collect_data(self, envs, agent, seed: int, device: torch.device):
        total_timesteps = len(self.obs)
        agent = deepcopy(agent).eval()
        fabric.seed_everything(seed=seed)
        next_obs = torch.tensor(
            envs.reset(
                seed=[seed + i for i in range(len(envs.envs))])[0],
            device=device)
        for _ in (pbar := trange(total_timesteps, leave=False)):
            with torch.no_grad():
                action, output, _, _, value = agent.get_action_and_value(
                    next_obs)
                value = value.squeeze(1).cpu()

            action = action.numpy(force=True)
            self.add(obs=next_obs.cpu(),
                     output=output.cpu(),
                     value=value)

            next_obs, _, _, _, _ = envs.step(action)
            next_obs = torch.tensor(next_obs, device=device)

            pbar.refresh()

    def get_dataset(self, envs):
        if hasattr(self, "dataset"):
            return self.dataset

        if self.full:
            pos = self.buffer_size
        else:
            pos = self.pos
        self.dataset = RolloutDataset(
            envs, self.obs[:pos], self.output[:pos], self.value[:pos]
        )
        return self.dataset


def gather_data(args, data_path, agent, device: torch.device):
    # Data gathering
    if not data_path.exists():
        # train
        train_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, 1_001, 0, False, None)]
        )
        train_b = RolloutBuffer(train_envs, args.train_bs)
        train_b.collect_data(train_envs, agent, 1_001, device)
        # val
        val_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, 1_101, 0, False, None)]
        )
        val_b = RolloutBuffer(val_envs, args.val_bs)
        val_b.collect_data(val_envs, agent, 1_101, device)
        # test https://www.random.org/ generate from between 1 - 10_000
        test_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, 1_639, 0, False, None)]
        )
        test_b = RolloutBuffer(test_envs, args.test_bs)
        test_b.collect_data(test_envs, agent, 1_639, device)

        with data_path.open("wb") as f:
            joblib.dump((train_b, val_b, test_b), f)
    with data_path.open("rb") as f:
        train_b, val_b, test_b = joblib.load(f)

    return train_b, val_b, test_b
