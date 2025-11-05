# Permission to use and modify granted by Eoin Kenny
# The code is from https://github.com/EoinKenny/Prototype-Wrapper-Network-ICLR23

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.agents.ppo import Agent
from src.env import ACTION_MAPPER, make_env
import gymnasium as gym
from pathlib import Path
import joblib
from tqdm import trange
import statistics
from lightning.fabric.loggers import TensorBoardLogger
from lightning import Fabric
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tyro
from dataclasses import dataclass
from torch.distributions import Categorical
from rl_utils import evaluate
from copy import deepcopy


@dataclass
class Args:
    env_id: str = "CarRacing-v2"
    seed: int = 0
    batch_size: int = 64
    latent_size: int = 512
    prototype_size: int = 50
    num_workers: int = 8
    lr: float = 1e-3
    weight_decay: float = 5e-4
    gamma: float = 0.98
    num_epochs: int = 1_000
    num_envs: int = 1
    patience: int = 10
    val_interval: int = 10


class ListModule(object):
    # Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class PWNet(nn.Module):

    def __init__(self, env_id: str, latent_size: int, prototype_size: int):
        super(PWNet, self).__init__()
        self.prototype_size = prototype_size

        self.env_id = env_id
        if self.env_id == "CarRacing-v2":
            self.num_classes = 5
            self.num_prototypes = 4
        elif self.env_id == "PongNoFrameskip-v4":
            self.num_classes = 6
            self.num_prototypes = 6
        elif self.env_id == "BreakoutNoFrameskip-v4":
            self.num_classes = 4
            self.num_prototypes = 4
        elif self.env_id == "MsPacmanNoFrameskip-v4":
            self.num_classes = 9
            self.num_prototypes = 9
        else:
            raise NotImplementedError

        self.ts = ListModule(self, 'ts_')
        for i in range(self.num_prototypes):
            transformation = nn.Sequential(
                nn.Linear(latent_size, prototype_size),
                nn.InstanceNorm1d(prototype_size),
                nn.ReLU(),
                nn.Linear(prototype_size, prototype_size),
            )
            self.ts.append(transformation)

        self.epsilon = 1e-5
        self.linear = nn.Linear(self.num_prototypes,
                                self.num_classes, bias=False)
        self.__make_linear_weights()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.nn_human_x = nn.Parameter(torch.randn(
            self.num_prototypes, latent_size), requires_grad=False)

    @staticmethod
    def get_latent(agent, x):
        return agent.network(x / 255.0)

    def __make_linear_weights(self):
        """
        Must be manually defined to connect prototypes to human-friendly concepts
        For example, -1 here corresponds to steering left, whilst the 1 below it to steering right
        Together, they can encapsulate the overall concept of steering
        More could be connected, but we just use 2 here for simplicity.
        """
        custom_weight_matrix = torch.eye(self.num_classes)
        if self.env_id == "CarRacing-v2":
            custom_weight_matrix = custom_weight_matrix[:,
                                                        :self.num_prototypes]

        self.linear.weight.data.copy_(custom_weight_matrix)

    def __proto_layer_l2(self, x, p):
        # output = list()
        b_size = x.shape[0]
        p = p.view(1, self.prototype_size).tile(b_size, 1)
        c = x.view(b_size, self.prototype_size)
        l2s = ((c - p)**2).sum(axis=1)
        act = torch.log((l2s + 1.) / (l2s + self.epsilon))
        return act

    def get_similarity(self, x):
        # Get the latent prototypes by putting them through the individual transformations
        trans_nn_human_x = list()
        for i, t in enumerate(self.ts):
            trans_nn_human_x.append(
                t(self.nn_human_x[i].view(1, -1)))
        latent_protos = torch.cat(trans_nn_human_x, dim=0)

        # Do similarity of inputs to prototypes
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append(self.__proto_layer_l2(
                t(x), action_prototype).view(-1, 1))
        p_acts = torch.cat(p_acts, axis=1)
        return p_acts

    def forward(self, x):

        # Get the latent prototypes by putting them through the individual transformations
        trans_nn_human_x = list()
        for i, t in enumerate(self.ts):
            trans_nn_human_x.append(
                t(self.nn_human_x[i].view(1, -1)))
        latent_protos = torch.cat(trans_nn_human_x, dim=0)

        # Do similarity of inputs to prototypes
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append(self.__proto_layer_l2(
                t(x), action_prototype).view(-1, 1))
        p_acts = torch.cat(p_acts, axis=1)

        # Put though activation function method
        logits = self.linear(p_acts)

        return logits


def main():
    args = tyro.cli(Args)

    def action_fn(x, a):
        if args.env_id != "CarRacing-v2":
            return torch.argmax(a(a.get_latent(agent, x)), -1)
        else:
            return Categorical(logits=a(a.get_latent(agent, x))).sample()

    def get_project_folder() -> Path:
        return Path(__file__).parent.parent

    folder = get_project_folder() / f"runs/train/ppo__{args.env_id}/version_0"
    pw_folder = get_project_folder() / f"runs/pwnet/{args.env_id}"
    if not pw_folder.exists():
        pw_folder.mkdir(parents=True)

    Fabric.seed_everything(args.seed)
    logger = TensorBoardLogger(root_dir=pw_folder, name="")
    logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    fabric = Fabric(loggers=logger)
    fabric.launch()

    def get_latent(agent, x):
        return agent.network(x / 255.0)

    def evaluate_loader(agent, model, loader, loss):
        model.eval()
        total_error = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                imgs, labels, _ = data
                imgs, labels = imgs.to(fabric.device), labels.to(fabric.device)
                imgs = get_latent(agent, imgs)
                labels = torch.argmax(labels, dim=1)

                logits = model(imgs)
                current_loss = loss(logits, labels)
                total_error += current_loss.item()
                total += len(imgs)
        model.train()
        return total_error / total

    def evaluate_loader_accuracy(agent, model, loader):
        model.eval()
        total_error = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                imgs, labels, _ = data
                imgs, labels = imgs.to(fabric.device), labels.to(fabric.device)
                imgs = get_latent(agent, imgs)
                labels = torch.argmax(labels, dim=1)

                logits = model(imgs)
                current_loss = (torch.argmax(logits, dim=1) == labels).sum()
                total_error += current_loss.item()
                total += len(imgs)
        model.train()
        return total_error / total

    # Start Collecting Data To Form Final Mean and Standard Error Results

    # My Code ---------------------

    # Setup env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, False, None)
         for i in range(args.num_envs)]
    )

    # Setup agent
    agent_params = torch.load(folder / "state.ckpt",
                              map_location=torch.device('cpu'))["model"]
    agent = Agent(envs)
    agent.load_state_dict(agent_params)
    agent.to(fabric.device).eval()

    # Loading data
    data = joblib.load(folder / "data.joblib")
    train = data[0].get_dataset(envs)
    val = data[1].get_dataset(envs)

    # # Generate images
    # img_folder = get_project_folder() / f"temp_{args.env_id}"
    # if not img_folder.exists():
    #     img_folder.mkdir()

    # for idx in range(5_000):
    #     obs = train[idx][0].numpy(force=True)
    #     probs = F.softmax(train[idx][1], -1)
    #     action_idx = torch.argmax(probs).item()
    #     action = ACTION_MAPPER[args.env_id][action_idx]

    #     fig, ax = plt.subplots(layout="constrained", figsize=(4, 4.25))

    #     title = f"idx={idx}__action={action}__prob={probs[action_idx]:.2f}.png"
    #     ax.imshow(obs[0], cmap='gray', vmin=0, vmax=255)
    #     ax.axis("off")
    #     fig.savefig(
    #         img_folder / title,
    #         bbox_inches='tight', pad_inches=0
    #     )

    #     plt.close()
    # raise

    train_loader = DataLoader(
        train, shuffle=True, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        val, shuffle=False, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers)

    # Human defined Prototypes for interpretable model (these were gotten manually earlier)
    # A clustering analysis could be done to help guide the search, or they can be manually chosen.
    # Lastly, they could also be learned as pwnet* shows in the comparitive tests
    if args.env_id == "CarRacing-v2":
        # do nothing=1983, steer left=1865, steer right=2740, gas=1246
        p_idxs = np.array([1983, 1865, 2740, 1246])
    elif args.env_id == "PongNoFrameskip-v4":
        # TODO: Find the prototypes
        # do nothing=36, do nothing (fire)=124, go up=369, go down=1120, go up (and fire)=434, go down(and fire)=196
        p_idxs = np.array([36, 124, 369, 1120, 434, 196])
    elif args.env_id == "BreakoutNoFrameskip-v4":
        # TODO: Find the prototypes
        # do nothing=44, do nothing (fire)=146, go right=111, go left=304
        p_idxs = np.array([44, 146, 111, 304])
    elif args.env_id == "MsPacmanNoFrameskip-v4":
        # TODO: Find the prototypes
        # noop=69, up=115, right=4691, left=1596, down=63, upright=251, upleft=668, downright=306, downleft=191
        p_idxs = np.array(
            [69, 115, 4691, 1596, 63, 251, 668, 306, 191])
    else:
        raise NotImplementedError()

    nn_human_x = get_latent(
        agent, train[p_idxs.flatten()][0].to(fabric.device))
    # nn_human_actions = torch.argmax(train[p_idxs.flatten()][1], dim=1)

    # Training
    model = PWNet(args.env_id, args.latent_size,
                  args.prototype_size).to(fabric.device).eval()
    model.nn_human_x.data.copy_(torch.tensor(nn_human_x))

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.gamma)
    best_error = float('inf')
    model.train()

    # Freeze Linear Layer
    model.linear.weight.requires_grad = False
    patience = 0

    for epoch in trange(args.num_epochs, desc="Training PWNet"):
        fabric.log("charts/lr", scheduler.get_last_lr()[0], epoch)
        running_loss = 0
        model.train()

        for instances, labels, _ in train_loader:
            optimizer.zero_grad()

            instances, labels = instances.to(
                fabric.device), labels.to(fabric.device)
            instances = get_latent(agent, instances)
            labels = torch.argmax(labels, dim=1)

            logits = model(instances)
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        fabric.log("train/loss", running_loss / len(train_loader), epoch)

        if (epoch + 1) % args.val_interval == 0:
            model.eval()

            returns, _ = evaluate(args.env_id, model, 10,
                                  action_fn, fabric.device, 6434,
                                  seed_everything=False)
            return_mean = statistics.mean(returns)
            val_error = evaluate_loader(agent, model, val_loader, ce_loss)
            accuracy = evaluate_loader_accuracy(agent, model, val_loader)

            fabric.log("val/accuracy", accuracy, epoch)
            fabric.log("val/return", return_mean, epoch)
            fabric.log("val/loss", val_error, epoch)

            prev_best = deepcopy(best_error)
            if -return_mean <= best_error:
                torch.save(model.state_dict(), Path(
                    logger.log_dir) / "pw_net.pth")
                best_error = -return_mean

            if -return_mean >= prev_best:
                patience += 1
                if patience >= args.patience:
                    break
            else:
                patience = 0
        scheduler.step()


if __name__ == "__main__":
    main()
