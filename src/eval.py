import gymnasium as gym
from src.env import make_env
from src.agents.ppo import Agent
from lightning.fabric import Fabric
from src.rl_utils import get_project_folder, evaluate, evaluate_acc
import statistics
import torch
import joblib
from src.buffer import gather_data
# Need for loading model
from src.main import Model, ProtoConcept
from src.pw_net import PWNet
from argparse import Namespace
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions import Categorical
from src.sdt import SDT
import pandas as pd
from copy import deepcopy
import tyro
from dataclasses import dataclass
from typing import Literal


def compute_accuracy(dataloader, action_pred_func):
    accs, weights = [], []
    for batch in dataloader:
        log_probs = batch[1]

        best_action = torch.argmax(log_probs, dim=1)
        worst_action = torch.argmin(log_probs, dim=1)
        best_log_prob = log_probs[torch.arange(len(log_probs)), best_action]
        worst_log_prob = log_probs[torch.arange(len(log_probs)), worst_action]

        sample_weight = best_log_prob - worst_log_prob
        action_true = torch.argmax(batch[1], dim=1).numpy(force=True)
        action_pred = action_pred_func(batch[0]).numpy(force=True)
        accs.append(action_true == action_pred)
        weights.append(sample_weight.numpy(force=True))
    accs = np.concatenate(accs)
    weights = np.concatenate(weights)

    adj_acc = np.sum(accs * weights) / weights.sum()
    acc = accs.sum() / len(accs)

    return acc, adj_acc


def setup_dataloader(fabric, train, val, test, envs):
    batch_size = 256
    num_workers = 8
    training_loader = DataLoader(train.get_dataset(envs),
                                 batch_size=batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=num_workers)
    val_loader = DataLoader(val.get_dataset(envs),
                            batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test.get_dataset(envs),
                             batch_size=batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=num_workers)
    training_loader, val_loader, test_loader = fabric.setup_dataloaders(
        training_loader, val_loader, test_loader)

    return training_loader, val_loader, test_loader


def setup(method, env_id, num_envs: int, version: int = 0,  seed: int = 0,
          loggers=None):
    Fabric.seed_everything(seed)
    fabric = Fabric(loggers=loggers)
    fabric.launch()

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, False, None)
            for i in range(num_envs)]
    )

    agent = Agent(envs)
    model_folder = get_project_folder() / \
        f"runs/train/ppo__{env_id}/version_0"
    state = fabric.load(model_folder / "state.ckpt")
    agent.load_state_dict(state["model"])
    agent.to(fabric.device)
    agent.eval()

    if "moe" in method:
        model_folder = get_project_folder() / \
            f"runs/clf/{env_id}/version_{version}"
        agent_path = model_folder / "state.ckpt"
        loaded_model = joblib.load(agent_path)
        model = sorted(loaded_model, key=lambda x: x[1])[-1][0]
    elif method == "pwnet":
        model_folder = get_project_folder() / \
            f"runs/pwnet/{env_id}/version_{version}"
        agent_path = model_folder / "pw_net.pth"
        model = PWNet(env_id, 512, 50)
        model.load_state_dict(torch.load(agent_path))
        model.to(fabric.device).eval()
    elif method == "tree":
        model_folder = get_project_folder() / \
            f"runs/sdt/{env_id}/version_{version}"
        agent_path = model_folder / "tree.pt"
        model = SDT(28224,
                    {"CarRacing-v2": 5, "PongNoFrameskip-v4": 6,
                     "BreakoutNoFrameskip-v4": 4, "MsPacmanNoFrameskip-v4": 9}[env_id],
                    fabric.device, 5)
        model.load_state_dict(torch.load(agent_path))
        model.to(fabric.device).eval()
    else:
        model = agent
    model.eval()

    train_b, val_b, test_b = gather_data(
        Namespace(train_bs=50_000, val_bs=50_000,
                  test_bs=50_000, env_id=env_id),
        get_project_folder() /
        f"runs/train/ppo__{env_id}/version_0/data.joblib",
        agent, fabric.device
    )
    train_dataloader, val_dataloader, test_dataloader = setup_dataloader(
        fabric, train_b, val_b, test_b, envs
    )

    return agent, model, envs, fabric, (train_dataloader, val_dataloader, test_dataloader), model_folder


@dataclass
class Args:
    method: Literal["moe", "tree", "black-box", "pwnet", "moe-kmeans"]
    version: int
    env_id: Literal["CarRacing-v2", "PongNoFrameskip-v4",
                    "BreakoutNoFrameskip-v4", "MsPacmanNoFrameskip-v4"]


# https://www.random.org/ generate from between 1 - 10_000
def main(n_sim: int = 10, num_envs: int = 1, seed: int = 8_395):
    args = tyro.cli(Args)
    method = args.method
    version = args.version
    env_id = args.env_id

    agent, model, envs, fabric, (train_dataloader, val_dataloader,
                                 test_dataloader), save_folder = setup(method, env_id, num_envs, version=version,
                                                                       seed=seed)

    if "moe" in method:
        if hasattr(model.clf, "get_n_leaves"):
            print(f"Number of leaf nodes={model.clf.get_n_leaves()}")

        def action(x, a):
            if env_id != "CarRacing-v2":
                return a.predict(x, False)
            else:
                return a.predict(x, True)

        def max_action(x, a):
            return a.predict(x, False)
        returns, lengths = evaluate(env_id, model, n_sim,
                                    action, fabric.device, seed)
        acc, adj_acc = evaluate_acc(env_id, agent, model, n_sim,
                                    max_action, fabric.device, seed)
    elif method == "pwnet":

        def action(x, a):
            if env_id != "CarRacing-v2":
                return torch.argmax(a(a.get_latent(agent, x)), -1)
            else:
                return Categorical(logits=a(a.get_latent(agent, x))).sample()

        def max_action(x, a):
            return torch.argmax(a(a.get_latent(agent, x)), -1)

        returns, lengths = evaluate(env_id, model, n_sim,
                                    action, fabric.device, seed)
        acc, adj_acc = evaluate_acc(env_id, agent, model, n_sim,
                                    max_action, fabric.device, seed)
    elif method == "tree":
        def action(x, a):
            if env_id != "CarRacing-v2":
                return torch.argmax(a(x / 255.0), -1)
            else:
                return Categorical(logits=a(x / 255.0)).sample()

        def max_action(x, a):
            return torch.argmax(a(x / 255.0), -1)

        returns, lengths = evaluate(env_id, model, n_sim,
                                    action, fabric.device, seed)
        acc, adj_acc = evaluate_acc(env_id, agent, model, n_sim,
                                    action, fabric.device, seed)

    else:
        returns, lengths = evaluate(env_id, agent, n_sim,
                                    lambda x, a: a.get_action_and_value(x)[0],
                                    fabric.device, seed)

        acc, adj_acc = evaluate_acc(env_id, agent, agent, n_sim,
                                    lambda x, a: torch.argmax(
                                        a.get_action_and_value(x)[1], dim=1),
                                    fabric.device, seed)

    df = pd.DataFrame.from_dict({
        "return": returns,
        "accuracy": acc,
        "adjusted_accuracy": adj_acc
    })

    df.to_csv(save_folder / "result.csv", index=False)

    print(
        f"Overview: method={method}, env={env_id}, num_envs={num_envs}, version={version}, n_sim={n_sim}")
    print(
        f"Return: mean={statistics.mean(returns)}, std={statistics.stdev(returns)}")
    print(
        f"Accuracy: vanillia={statistics.mean(acc)}, adjusted={statistics.mean(adj_acc)}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
