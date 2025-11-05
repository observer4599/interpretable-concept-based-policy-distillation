import gymnasium as gym
from src.env import make_env
from src.agents.ppo import Agent
from lightning.fabric import Fabric
from src.rl_utils import get_project_folder
import torch
import joblib
from src.buffer import gather_data
from src.main import Model, ProtoConcept  # Need for loading model
from argparse import Namespace
from src.pw_net import PWNet
import numpy as np
from src.eval import setup_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import warnings
from src.sdt import SDT
from src.saliency import pertubation_saliency
from torchvision.transforms import GaussianBlur
import tyro
from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:
    mode: Literal["blur", "mean", "zero"] = "blur"


def compute_importance(fabric, model, dataloader, n_samples: int = 2_500):
    model.eval()
    idx = np.random.choice(np.arange(len(dataloader.dataset)), n_samples)

    attr = pertubation_saliency(dataloader.dataset.obs[idx].to(fabric.device),
                                model, fabric.device)

    attr = np.expand_dims(attr, 1)
    return attr, idx


def compute_error(input_probs: torch.Tensor, target_probs: torch.Tensor):
    diff = 0.5 * (input_probs - target_probs) ** 2
    return torch.sum(diff, -1).numpy(force=True)


def compute_errors(fabric, models, dataloader, attr, idx, mode):
    errors = defaultdict(list)

    if mode == "mean":
        print("Mode: mean")
        mask_val = dataloader.dataset.obs.mean().item()
    elif mode == "blur":
        imges = dataloader.dataset.obs[idx]
        blur_transform = GaussianBlur((11, 11), 3)
        mask_val = blur_transform(imges).to(fabric.device)
    else:
        raise NotImplementedError()
    attr_tensor = torch.tensor(attr, device=fabric.device)
    for q in tqdm(np.arange(0, 110, 10)):
        val = np.percentile(attr, q=q, axis=(1, 2, 3), keepdims=True)
        val = torch.tensor(val, device=fabric.device)
        buffer = defaultdict(list)

        obs = dataloader.dataset.obs[idx].to(fabric.device)
        masked_obs = torch.where(
            attr_tensor >= val if q != 100 else attr_tensor > val, obs, mask_val)

        buffer["agent"].append(compute_error(
            F.softmax(models["agent"](masked_obs), -1),
            F.softmax(models["agent"](obs), -1)
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buffer["pwnet"].append(compute_error(
                F.softmax(models["pwnet"](
                    models["pwnet"].get_latent(models["agent"], masked_obs)), -1),
                F.softmax(models["pwnet"](
                    models["pwnet"].get_latent(models["agent"], obs)), -1)
            ))

        buffer["moe"].append(compute_error(
            F.softmax(models["moe"](masked_obs), -1),
            F.softmax(models["moe"](obs), -1)
        ))
        buffer["viper"].append(compute_error(
            models["viper"](masked_obs),
            models["viper"](obs)
        ))
        buffer["sdt"].append(compute_error(
            F.softmax(models["sdt"](masked_obs / 255.0), -1),
            F.softmax(models["sdt"](obs / 255.0), -1)
        ))

        for key, value in buffer.items():
            errors[key].append(np.concatenate(value))

    return errors


def setup(env_id, num_envs: int = 10, seed: int = 0):
    Fabric.seed_everything(seed)
    fabric = Fabric()
    fabric.launch()
    models = {}

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, False, None)
            for i in range(num_envs)]
    )

    runs_folder = get_project_folder() / "runs"

    # agent
    agent = Agent(envs)
    state = fabric.load(runs_folder /
                        f"train/ppo__{env_id}/version_0/state.ckpt")
    agent.load_state_dict(state["model"])
    models["agent"] = agent
    # moe
    moe = joblib.load(runs_folder /
                      f"clf/{env_id}/version_0/state.ckpt")
    moe = sorted(moe, key=lambda x: x[1])[-1][0]
    models["moe"] = moe
    # viper
    viper = joblib.load(runs_folder /
                        f"clf/{env_id}/version_1/state.ckpt")
    viper = sorted(viper, key=lambda x: x[1])[-1][0]
    models["viper"] = viper
    # pwnet
    pwnet = PWNet(env_id, 512, 50)
    pwnet.load_state_dict(torch.load(runs_folder /
                                     f"pwnet/{env_id}/version_0/pw_net.pth"))
    models["pwnet"] = pwnet
    # sdt
    sdt = SDT(28224,
              {"CarRacing-v2": 5, "PongNoFrameskip-v4": 6,
                  "BreakoutNoFrameskip-v4": 4, "MsPacmanNoFrameskip-v4": 9}
              [env_id], fabric.device, 5)
    sdt.load_state_dict(torch.load(runs_folder /
                                   f"sdt/{env_id}/version_0/tree.pt"))
    models["sdt"] = sdt

    for model in models.values():
        model.to(fabric.device)
        model.eval()

    train_b, val_b, test_b = gather_data(
        Namespace(train_b=50_000, val_b=50_000, test_b=50_000, env_id=env_id),
        runs_folder /
        f"train/ppo__{env_id}/version_0/data.joblib",
        agent, fabric.device
    )
    train_dataloader, val_dataloader, test_dataloader = setup_dataloader(
        fabric, train_b, val_b, test_b, envs
    )

    return models, envs, fabric, (train_dataloader, val_dataloader, test_dataloader)


def main():
    args = tyro.cli(Args)
    info = {"mode": args.mode}

    for env_id in ["CarRacing-v2", "PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "MsPacmanNoFrameskip-v4"]:
        attr_file = get_project_folder(
        ) / f"runs/train/ppo__{env_id}/version_0/attr.joblib"

        models, _, fabric, (_, _, test_dataloader) = setup(env_id)
        if attr_file.exists():
            attr, idx = joblib.load(attr_file)
        else:
            attr, idx = compute_importance(
                fabric, models["agent"], test_dataloader)
            joblib.dump((attr, idx), attr_file)

        info[env_id] = compute_errors(
            fabric, models, test_dataloader, attr, idx, args.mode)

    return info


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    INFO = main()
