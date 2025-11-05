# Some of the code in this file is from cleanRL. The license is in ppo.py

from pathlib import Path
import numpy as np
import torch
from lightning.fabric import Fabric
import gymnasium as gym
from src.env import make_env
from collections import defaultdict


# Code from https://github.com/numpy/numpy/issues/15201
def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def get_project_folder() -> Path:
    return Path(__file__).parent.parent


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate(env_id, test_model, eval_episodes: int, action_func,
             device: torch.device, seed: int = 2_001, num_envs: int = 1,
             seed_everything: bool = True):
    if seed_everything:
        Fabric.seed_everything(seed=seed)
    test_model.eval()

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, False, None)
            for i in range(num_envs)]
    )

    next_obs = torch.tensor(envs.reset(
        seed=[seed + i for i in range(len(envs.envs))]
    )[0], device=device)
    episodic_returns, episodic_lengths = [], []

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            action = action_func(next_obs, test_model).numpy(force=True)

        next_obs, _, _, _, infos = envs.step(action)
        next_obs = torch.tensor(next_obs, device=device)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None or "episode" not in info:
                continue
            episodic_returns.append(info["episode"]["r"][0])
            episodic_lengths.append(info["episode"]["l"][0])

    del envs
    del next_obs
    del action
    return episodic_returns, episodic_lengths


def compute_accuracy(logits, action_pred):
    log_probs = logits

    best_action = np.argmax(log_probs, axis=1)
    worst_action = np.argmin(log_probs, axis=1)
    best_log_prob = log_probs[np.arange(len(log_probs)), best_action]
    worst_log_prob = log_probs[np.arange(len(log_probs)), worst_action]

    sample_weight = best_log_prob - worst_log_prob
    action_true = np.argmax(logits, axis=1)
    accs = action_true == action_pred
    weights = sample_weight

    adj_acc = np.sum(accs * weights) / weights.sum()
    acc = accs.sum() / len(accs)

    return acc, adj_acc


def evaluate_acc(env_id, agent, test_model, eval_episodes: int, action_func,
                 device: torch.device, seed: int = 2_001, num_envs: int = 1,
                 seed_everything: bool = True):
    if seed_everything:
        Fabric.seed_everything(seed=seed)
    agent.eval()
    test_model.eval()

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, False, None)
            for i in range(num_envs)]
    )

    next_obs = torch.tensor(envs.reset(
        seed=[seed + i for i in range(len(envs.envs))]
    )[0], device=device)
    storage = defaultdict(list)

    while len(storage["episodic_returns"]) < eval_episodes:
        with torch.no_grad():
            action, logits, _, _, _ = agent.get_action_and_value(
                next_obs)
            action = action.numpy(force=True)

            storage["logits"].append(logits.numpy(force=True))
            storage["action"].append(action_func(
                next_obs, test_model).numpy(force=True))

        next_obs, _, _, _, infos = envs.step(action)
        next_obs = torch.tensor(next_obs, device=device)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None or "episode" not in info:
                continue

            acc, adj_acc = compute_accuracy(
                np.concatenate(storage["logits"], axis=0),
                np.concatenate(storage["action"], axis=0)
            )

            storage["acc"].append(acc)
            storage["adj_acc"].append(adj_acc)
            storage["logits"].clear()
            storage["action"].clear()

            storage["episodic_returns"].append(info["episode"]["r"][0])
            storage["episodic_lengths"].append(info["episode"]["l"][0])

    del envs
    del next_obs
    del action
    return storage["acc"], storage["adj_acc"]
