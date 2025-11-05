# BSD 3-Clause License

# Copyright (c) 2018 - 2021 Yi-Xuan Xu.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tyro
from dataclasses import dataclass
import gymnasium as gym
from src.rl_utils import get_project_folder
from src.env import make_env
import torch
import torch.nn as nn
from src.agents.ppo import Agent
import joblib
from torch.utils.data import DataLoader
from lightning.fabric.loggers import TensorBoardLogger
from lightning import Fabric
import math
import torch.nn.functional as F
import torchmetrics
from tqdm import trange
from pathlib import Path
from rl_utils import evaluate
from torch.distributions import Categorical
import statistics
from copy import deepcopy


class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            device,
            depth=5,
            lamda=1e-3):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda
        self.device = device

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)

    def forward(self, X, is_training_data=False):
        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0, device=self.device)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0, device=self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / (torch.sum(_mu[:, node // 2], dim=0) + 1e-8)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * \
                (torch.log(alpha + 1e-8) + torch.log(1 - alpha + 1e-8))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1, device=self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))


@dataclass
class Args:
    env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 0
    num_envs: int = 8
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 2.5e-4
    num_epochs: int = 5_000
    depth: int = 5
    penalty: float = 1e-2
    val_interval: int = 10
    weight_decay: float = 1e-4
    patience: int = 5


def setup():
    args = tyro.cli(Args)
    Fabric.seed_everything(args.seed)
    folder = get_project_folder() / f"runs/train/ppo__{args.env_id}/version_0"

    sdt_folder = get_project_folder() / f"runs/sdt/{args.env_id}"

    logger = TensorBoardLogger(root_dir=sdt_folder, name="")
    logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    fabric = Fabric(loggers=logger)
    fabric.launch()

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
    agent = fabric.setup(agent).eval()

    # Loading data
    data = joblib.load(folder / "data.joblib")
    train = data[0].get_dataset(envs)
    val = data[1].get_dataset(envs)

    train_loader = DataLoader(
        train, shuffle=True, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        val, shuffle=False, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers)

    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader)

    tree = SDT(
        math.prod(list(train.obs.shape[1:])), train.output.shape[1],
        fabric.device, args.depth, args.penalty)
    optimizer = torch.optim.Adam(
        tree.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tree, optimizer = fabric.setup(tree, optimizer)
    metrics = {"train_loss": torchmetrics.aggregation.MeanMetric(),
               "train_penalty": torchmetrics.aggregation.MeanMetric(),
               "train_accuracy": torchmetrics.classification.MulticlassAccuracy(train.output.shape[1]),
               "val_loss": torchmetrics.aggregation.MeanMetric(),
               "val_accuracy": torchmetrics.classification.MulticlassAccuracy(train.output.shape[1])}

    best_error = float('inf')
    patience = 0

    def action(x, a):
        if args.env_id != "CarRacing-v2":
            return torch.argmax(a(x / 255.0), -1)
        else:
            return Categorical(logits=a(x / 255.0)).sample()
    for epoch in trange(args.num_epochs):
        tree.train()
        for batch in train_loader:
            input, target, _ = batch
            input = input.view(len(input), -1)

            optimizer.zero_grad()
            output, penalty = tree(input / 255.0, is_training_data=True)
            loss = F.kl_div(F.log_softmax(output, -1),
                            F.log_softmax(target, -1),
                            log_target=True,
                            reduction="batchmean")
            loss += penalty
            fabric.backward(loss)
            optimizer.step()

            metrics["train_loss"](loss.item())
            metrics["train_penalty"](penalty.item())
            metrics["train_accuracy"](torch.argmax(output, -1).detach().cpu(),
                                      torch.argmax(target, -1).detach().cpu())
        fabric.log("train_loss", metrics["train_loss"].compute(), epoch)
        fabric.log("train_penalty", metrics["train_penalty"].compute(), epoch)
        fabric.log("train_accuracy",
                   metrics["train_accuracy"].compute(), epoch)

        if (epoch + 1) % args.val_interval == 0:
            tree.eval()

            returns, _ = evaluate(args.env_id, tree, 10,
                                  action,
                                  agent.network[0].weight.device,
                                  seed=6434, seed_everything=False)
            return_mean = statistics.mean(returns)
            for batch in val_loader:
                input, target, _ = batch
                input = input.view(len(input), -1)
                output = tree(input / 255.0)
                val_loss = F.kl_div(F.log_softmax(output, -1),
                                    F.log_softmax(target, -1),
                                    log_target=True,
                                    reduction="batchmean")
                metrics["val_loss"](val_loss.item())
                metrics["val_accuracy"](torch.argmax(output, -1).detach().cpu(),
                                        torch.argmax(target, -1).detach().cpu())
            val_loss = -return_mean

            prev_best = deepcopy(best_error)
            if val_loss <= best_error:
                best_error = val_loss
                torch.save(tree.state_dict(), Path(
                    logger.log_dir) / "tree.pt")

            fabric.log("val_return", return_mean, epoch)
            fabric.log("val_return_std", statistics.stdev(returns), epoch)
            fabric.log("val_loss", metrics["val_loss"].compute(), epoch)
            fabric.log("val_accuracy",
                       metrics["val_accuracy"].compute(), epoch)

            if val_loss >= prev_best:
                patience += 1
                if patience >= args.patience:
                    break
            else:
                patience = 0

        for metric in metrics.values():
            metric.reset()


def main():
    setup()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
