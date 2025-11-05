# Some of the code in this file is from cleanRL. The license is in ppo.py
from src.agents.ppo import Agent
from lightning.fabric import Fabric
from src.rl_utils import get_project_folder, evaluate, categorical
import gymnasium as gym
import argparse
import torch
from lightning.fabric.loggers import TensorBoardLogger
from src.buffer import gather_data
from torch.utils.data import DataLoader
from pathlib import Path
from distutils.util import strtobool
from sklearn.cluster import KMeans
from src.env import make_env
import numpy as np
from torch import nn
from tqdm import trange, tqdm
import statistics
from collections import defaultdict
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import joblib
from lightning.fabric.wrappers import _FabricModule
from torch.utils.data import TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--agent-version", type=int, default=0)
    parser.add_argument("--load-version", type=int, default=-1)

    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers")
    parser.add_argument("--env-id", type=str, default="CarRacing-v2",
                        help="the id of the environment")
    parser.add_argument("--algo", type=str, default="ppo",
                        help="the algorithm of the policy")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")

    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--train-bs", type=int, default=100_000,
                        help="training set buffer size")
    parser.add_argument("--val-bs", type=int, default=50_000,
                        help="validation set buffer size")
    parser.add_argument("--test-bs", type=int, default=50_000,
                        help="testing set buffer size")
    parser.add_argument("--val-interval", type=int, default=10,
                        help="how often to run validation")
    parser.add_argument("--alpha", type=float, default=1,
                        help="")
    parser.add_argument("--n-concepts", type=int, default=4,
                        help="")
    parser.add_argument("--target-layer", type=int, default=3,
                        help="")
    parser.add_argument("--mode", type=str,
                        choices=["kmeans", "nmf",])

    parser.add_argument("--save-start", type=int)
    parser.add_argument("--max-patience", type=int)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--decay-factor", type=float, default=1.0,
                        help="")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="")
    parser.add_argument("--epochs", type=int, default=5_000,
                        help="")
    parser.add_argument("--reg-coef", type=float,
                        help="")
    parser.add_argument("--n-experts", type=int,
                        help="")

    # Classfier specific
    parser.add_argument("--clf-type", type=str,
                        choices=["none", "lr", "dt", "rf",
                                 "moe", "moe-kmeans"])
    parser.add_argument("--max-depth", type=int, default=None,
                        help="")
    parser.add_argument("--n-clf", type=int, default=10,
                        help="")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="")
    parser.add_argument("--ccp-alpha", type=float, default=0,
                        help="")
    parser.add_argument("--n-rollouts", type=int, help="")
    args = parser.parse_args()
    return args


def flat_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        hidden_flat = torch.swapaxes(hidden, 1, 3)
        return torch.reshape(hidden_flat, (-1, hidden_flat.shape[-1]))
    hidden_flat = np.swapaxes(hidden, 1, 3)
    return np.reshape(hidden_flat, (-1, hidden_flat.shape[-1]))


def unflat_hidden(hidden_flat, shape):
    shape = (shape[0], *shape[2:], shape[1])
    if isinstance(hidden_flat, torch.Tensor):
        hidden = torch.reshape(hidden_flat, shape)
        return torch.swapaxes(hidden, 1, 3)
    hidden = np.reshape(hidden_flat, shape)
    return np.swapaxes(hidden, 1, 3)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return min(slope * t + start_e, end_e)


class ProtoConcept(torch.nn.Module):
    def __init__(self, env_id, id_, input_dim, n_experts, n_actions, n_epochs: int,
                 reg_coef, batch_size, val_interval, max_patience, save_start,
                 kmeans_expert=None) -> None:
        super().__init__()
        self.env_id = env_id
        self.id_ = id_
        self.n_experts = n_experts
        self.n_actions = n_actions
        self.n_epochs = n_epochs
        self.reg_coef = reg_coef
        self.batch_size = batch_size
        self.max_patience = max_patience
        self.val_interval = val_interval
        self.save_start = save_start
        self.kmeans_expert = kmeans_expert

        self.hard = True

        if self.n_experts > 1:
            if kmeans_expert is None:
                self.transform = nn.Sequential(
                    nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
                    nn.ReLU(), nn.Linear(128, n_experts),
                )
            else:
                self.transform = self.kmeans_expert

        self.experts = nn.ModuleList([nn.Linear(input_dim, self.n_actions,
                                                bias=False)
                                      for _ in range(n_experts)])

    def device(self) -> torch.device:
        return self.experts[0].weight.device

    def forward(self, X):
        if self.n_experts > 1:
            if self.kmeans_expert is None:
                y_soft = F.softmax(self.transform(X), -1)

                if self.hard:
                    index = y_soft.max(-1, keepdim=True)[1]
                    y_hard = torch.zeros_like(
                        y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                    ret = y_hard - y_soft.detach() + y_soft
                else:
                    ret = y_soft
            else:
                concept_flat = self.transform.predict(X.numpy(force=True))

                index = np.zeros(
                    (concept_flat.size, len(self.transform.cluster_centers_)))
                index[np.arange(concept_flat.size), concept_flat] = 1.0
                ret = torch.tensor(index, device=X.device)

            output = torch.zeros((len(X), self.n_actions),
                                 device=X.device)
            for i, p in enumerate(self.experts):
                output = output + p(X) * ret[:, i:i + 1]

            return output, ret
        return self.experts[0](X), None

    def predict(self, X: torch.Tensor, numpy: bool = True):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=self.device())
        pred = torch.argmax(self(X)[0], dim=1)
        if numpy:
            return pred.numpy(force=True)
        return pred

    def predict_proba(self, X, numpy: bool = True):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=self.device())
        pred_proba = F.softmax(self(X)[0], -1)
        if numpy:
            return pred_proba.numpy(force=True)
        return pred_proba

    def fit(self, X, y, clf, val_dataloader, fabric, optimizer, scheduler):
        self.train()

        dataloader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            shuffle=True, num_workers=8, batch_size=self.batch_size
        )
        dataloader = fabric.setup_dataloaders(dataloader)
        err = float("inf")
        patience = 0

        for epoch in range(self.n_epochs):
            fabric.log("charts/lr", scheduler.get_last_lr()[0], epoch)
            log = defaultdict(list)
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                logits, _ = self(X_batch)

                pred_loss = F.kl_div(F.log_softmax(logits, -1),
                                     F.log_softmax(y_batch, -1),
                                     log_target=True,
                                     reduction="batchmean")
                loss = pred_loss
                log["prediction"].append(pred_loss.item())

                # Regularization
                reg_loss = None
                for expert_i in range(self.n_experts):
                    if reg_loss is None:
                        reg_loss = (1 / self.n_experts) * \
                            torch.mean(self.experts[expert_i].weight.abs())
                    else:
                        reg_loss += (1 / self.n_experts) * \
                            torch.mean(self.experts[expert_i].weight.abs())

                loss += self.reg_coef * reg_loss
                log["regularization"].append(reg_loss.item())

                fabric.backward(loss)
                optimizer.step()

            fabric.log(f"train-loss/{self.id_}",
                       statistics.mean(log["prediction"]), epoch)
            fabric.log(f"train-reg/{self.id_}",
                       statistics.mean(log["regularization"]), epoch)

            if (epoch + 1) % self.val_interval == 0:
                returns, _ = evaluate(clf.env_id, clf, 10,
                                      lambda x, a: a.predict(
                                          x, False if clf.env_id != "CarRacing-v2" else True),
                                      clf.agent.network[0].weight.device,
                                      seed=6434, seed_everything=False)
                return_mean = statistics.mean(returns)
                fabric.log(f"val-return/{self.id_}",
                           return_mean, epoch)
                fabric.log(f"val-return-std/{self.id_}",
                           statistics.stdev(returns), epoch)
                self.eval()
                loss_err, clf_err = 0, 0

                for X_batch, y_batch in val_dataloader:
                    logits, _ = self(X_batch)
                    loss_err += F.kl_div(F.log_softmax(logits, -1),
                                         F.log_softmax(y_batch, -1),
                                         log_target=True,
                                         reduction="batchmean").item()
                    clf_err += (torch.argmax(logits, -1) !=
                                torch.argmax(y_batch, -1)).float().mean().item()
                fabric.log(f"val-loss/{self.id_}",
                           loss_err / len(val_dataloader), epoch)
                fabric.log(f"val-clf/{self.id_}",
                           clf_err / len(val_dataloader), epoch)

                epoch_err = -return_mean

                # Choose the best parameters
                prev_best = deepcopy(err)
                if epoch + 1 > self.save_start and epoch_err <= err:
                    err = epoch_err
                    model_sd = deepcopy(self.state_dict())
                    optim_sd = deepcopy(optimizer.state_dict())

                if epoch_err >= prev_best:
                    patience += 1
                    if patience >= self.max_patience:
                        break
                else:
                    patience = 0
                self.train()

            scheduler.step()

        self.load_state_dict(model_sd)
        optimizer.load_state_dict(optim_sd)
        self.id_ += 1

        self.eval()
        return self


class Model(torch.nn.Module):
    def __init__(self, env_id, clf, agent, reducer, std) -> None:
        super().__init__()
        self.env_id = env_id
        self.clf = clf
        self.agent = agent
        self.reducer = reducer
        self.le = LabelEncoder()
        self.std = std

    @staticmethod
    def obs_to_concepts(std, X, agent, reducer, batch_size: int = 256):
        device = agent.network[0].weight.device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=device)
        concepts = []
        with torch.no_grad():
            for i in range(-(len(X) // -batch_size)):
                hidden, _ = agent.get_conv_latent(
                    X[i * batch_size:(i + 1) * batch_size].to(device),
                    agent.target_layer)

                hidden_flat = flat_hidden(hidden).numpy(force=True)

                if isinstance(reducer, NMF):
                    concept_flat = reducer.transform(hidden_flat)
                elif isinstance(reducer, KMeans):
                    concept_flat = reducer.predict(hidden_flat)

                    onehot = np.zeros(
                        (concept_flat.size, len(reducer.cluster_centers_)))
                    onehot[np.arange(concept_flat.size), concept_flat] = 1.0
                    concept_flat = onehot.astype(hidden_flat.dtype)
                else:
                    raise NotImplementedError()

                if not isinstance(reducer, _FabricModule):
                    shape = [-1, concept_flat.shape[1], *hidden.shape[2:]]
                    concept = unflat_hidden(concept_flat, shape)
                concepts.append(concept)

        concepts = np.concatenate(concepts, axis=0)

        return np.reshape(concepts, (-1, concepts[0].size)), shape, std

    def get_cluster(self, X):
        if len(X.shape) != 2:
            X, _, _ = self.obs_to_concepts(self.std,
                                           X, self.agent, self.reducer)
        return F.softmax(self.clf.transform(torch.tensor(X, device=self.clf.experts[0].weight.device)), -1)

    def predict(self, X, sample: bool = False):
        if len(X.shape) != 2:
            X, _, _ = self.obs_to_concepts(self.std,
                                           X, self.agent, self.reducer)
        if sample:
            probs = self.clf.predict_proba(X)
            action = categorical(probs)
        else:
            action = self.clf.predict(X)
        if isinstance(self.clf, _FabricModule):
            return torch.tensor(action)
        return torch.tensor(self.le.inverse_transform(action))

    def forward(self, X):
        if len(X.shape) != 2:
            X, _, _ = self.obs_to_concepts(self.std,
                                           X, self.agent, self.reducer)
        if isinstance(self.clf, _FabricModule):
            logits = self.clf(torch.tensor(
                X, device=self.agent.network[0].weight.device))[0]
            return logits
        return torch.tensor(self.clf.predict_proba(X))

    def label_transform(self, y):
        new_y = np.copy(y)
        return self.le.fit_transform(new_y)

    def fit(self, X, y_logits, idx, val_dataloader, fabric, optimizer, scheduler):
        if isinstance(self.clf, _FabricModule):
            self.clf.fit(X, y_logits, self,
                         val_dataloader, fabric, optimizer, scheduler)
        else:
            y = self.label_transform(np.argmax(y_logits, axis=1))
            self.clf.fit(X[idx], y[idx])
        return self


def setup_dataloader(info):
    training_loader = DataLoader(info["train_b"].get_dataset(info["envs"]),
                                 batch_size=info["args"].batch_size,
                                 shuffle=True, pin_memory=False,
                                 num_workers=info["args"].num_workers)
    val_loader = DataLoader(info["val_b"].get_dataset(info["envs"]),
                            batch_size=info["args"].batch_size,
                            shuffle=False, pin_memory=False,
                            num_workers=info["args"].num_workers)
    test_loader = DataLoader(info["test_b"].get_dataset(info["envs"]),
                             batch_size=info["args"].batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=info["args"].num_workers)
    training_loader, val_loader, test_loader = info["fabric"].setup_dataloaders(
        training_loader, val_loader, test_loader)

    return training_loader, val_loader, test_loader


def train(info, train_dataloader, val_dataloader):
    args = info["args"]
    Fabric.seed_everything(args.seed)
    agent = info["fabric"].setup(info["agent"]).eval()
    envs = info["envs"]

    X_train, shape, std = Model.obs_to_concepts(
        None,
        train_dataloader.dataset.obs,
        agent, info["reducer"])

    if args.clf_type == "none":
        return [(Model(args.env_id, None, agent, info["reducer"], std), 0)]

    X_val, _, _ = Model.obs_to_concepts(
        std, val_dataloader.dataset.obs, agent, info["reducer"])
    val_dataloader = DataLoader(
        TensorDataset(torch.tensor(X_val), val_dataloader.dataset.output),
        shuffle=False, num_workers=info["args"].num_workers,
        batch_size=info["args"].batch_size, pin_memory=True)

    obs = X_train
    logits = train_dataloader.dataset.output.numpy(
        force=True)

    val_dataloader = info["fabric"].setup_dataloaders(val_dataloader)

    next_obs = torch.tensor(envs.reset(
        seed=[args.seed + j for j in range(len(envs.envs))])[0],
        device=info["fabric"].device)
    clfs = []

    for i in trange(args.n_clf):
        returns = []

        if i != 0:
            while len(returns) < args.n_rollouts:
                # Add observation to buffer

                concept, shape, _ = Model.obs_to_concepts(std,
                                                          next_obs, agent, info["reducer"])

                if 'obs' in locals():
                    obs = np.concatenate((obs, concept), axis=0)
                else:
                    obs = concept
                action, logit, _, _, _ = agent.get_action_and_value(
                    next_obs)

                # Add output to buffer
                if 'logits' in locals():
                    logits = np.append(logits, logit.numpy(force=True), axis=0)
                else:
                    logits = logit.numpy(force=True)

                if i != 0:
                    action = clfs[-1][0].predict(next_obs, True)
                next_obs, _, _, _, infos = envs.step(action.numpy(force=True))
                next_obs = torch.tensor(next_obs, device=info["fabric"].device)

                # Only print when at least 1 env is done
                if "final_info" not in infos:
                    continue

                for ep_info in infos["final_info"]:
                    # Skip the envs that are not done
                    if ep_info is None or "episode" not in ep_info:
                        continue
                    returns.append(ep_info["episode"]["r"][0])

        log_probs = logits
        best_action = np.argmax(logits, axis=1)
        worst_action = np.argmin(logits, axis=1)
        best_log_prob = log_probs[np.arange(len(log_probs)), best_action]
        worst_log_prob = log_probs[np.arange(len(log_probs)), worst_action]

        sample_weight = best_log_prob - worst_log_prob
        sample_prob = sample_weight / sample_weight.sum()

        match args.clf_type:
            case "moe":
                main_clf = ProtoConcept(
                    args.env_id,
                    i, shape[1] * shape[2] * shape[3],
                    info["args"].n_experts, logits.shape[1],
                    info["args"].epochs, info["args"].reg_coef,
                    info["args"].batch_size, info["args"].val_interval,
                    info["args"].max_patience, info["args"].save_start)

                main_clf, optimizer = info["fabric"].setup(
                    main_clf, optim.Adam(
                        main_clf.parameters(),
                        lr=info["args"].learning_rate,
                        weight_decay=args.weight_decay))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=args.decay_factor)
            case "moe-kmeans":
                main_clf = ProtoConcept(
                    args.env_id,
                    i, shape[1] * shape[2] * shape[3],
                    info["args"].n_experts, logits.shape[1],
                    info["args"].epochs, info["args"].reg_coef,
                    info["args"].batch_size, info["args"].val_interval,
                    info["args"].max_patience, info["args"].save_start,
                    info["kmeans_expert"])

                main_clf, optimizer = info["fabric"].setup(
                    main_clf, optim.Adam(
                        main_clf.parameters(),
                        lr=info["args"].learning_rate,
                        weight_decay=args.weight_decay))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=args.decay_factor)

            case "dt":
                main_clf = DecisionTreeClassifier(random_state=args.seed,
                                                  ccp_alpha=args.ccp_alpha)
                optimizer = None
                scheduler = None
            case _:
                raise NotImplementedError("Not a valid model")

        clf = Model(args.env_id, main_clf, agent, info["reducer"], std)

        n_samples = min(args.n_samples, np.sum(sample_prob > 0))
        idx = np.random.choice(len(sample_prob), n_samples, p=sample_prob)

        clf.train()
        clf.fit(obs, logits, idx,
                val_dataloader, info["fabric"], optimizer, scheduler)
        clf.eval()
        returns, _ = evaluate(args.env_id, clf, 20,
                              lambda x, a: a.predict(
                                  x, False if args.env_id != "CarRacing-v2" else True), agent.network[0].weight.device,
                              seed=6434)
        clfs.append((deepcopy(clf), statistics.mean(returns)))
        clfs[-1][0].eval()

        info["fabric"].log("charts/episodic_return",
                           statistics.mean(returns), i)
        info["fabric"].log("charts/max_episodic_return", max(returns), i)
        info["fabric"].log("charts/min_episodic_return", min(returns), i)
        info["fabric"].log("charts/episodic_return_std",
                           statistics.stdev(returns), i)

        info["fabric"].log("charts/n_obs", len(obs), i)
        info["fabric"].log("charts/n_training_obs", n_samples, i)

        if isinstance(main_clf, DecisionTreeClassifier):
            info["fabric"].log("model/depth", main_clf.get_depth(), i)
            info["fabric"].log("model/n_leaves", main_clf.get_n_leaves(), i)

        joblib.dump(clfs, Path(
            info["fabric"].logger.experiment.log_dir) / "state.ckpt")

    return clfs


def learn_concepts(mode: str, agent, dataloader, n_concepts: int, n_experts: int, seed: int,
                   n_samples: int = 100_000):
    agent = agent.eval()

    x = []
    with torch.no_grad():
        for obs, _, _ in tqdm(dataloader):
            hidden, _ = agent.get_conv_latent(
                obs, agent.target_layer)
            x.append(hidden.numpy(force=True))
    x = np.concatenate(x, axis=0)
    x_flat = flat_hidden(x)

    idx = np.random.choice(np.arange(len(x_flat)), n_samples, replace=False)

    if "kmeans" in mode:
        nmf = KMeans(n_clusters=n_concepts, random_state=seed,
                     n_init="auto").fit(x_flat[idx])
    else:
        nmf = NMF(n_components=n_concepts, random_state=seed).fit(x_flat[idx])

    x_transformed = nmf.transform(flat_hidden(x))
    x_transformed = unflat_hidden(
        x_transformed, (-1, x_transformed.shape[1], x.shape[2], x.shape[3]))
    x_transformed = np.reshape(x_transformed, (x_transformed.shape[0], -1))
    k_means_expert = KMeans(n_clusters=n_experts, random_state=seed,
                            n_init="auto").fit(x_transformed)

    return nmf, k_means_expert


def setup():
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    Fabric.seed_everything(args.seed)

    log_dir = get_project_folder() / "runs/clf"
    log_name = str(args.env_id)

    if args.load_version >= 0:
        info = {"clf": joblib.load(
            log_dir / log_name / f"version_{args.load_version}/state.ckpt")}
        logger = None
    else:
        logger = TensorBoardLogger(
            root_dir=log_dir, name=log_name)
        logger.experiment.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        info = {}

    fabric = Fabric(loggers=logger)
    fabric.launch()

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, False, None)
            for i in range(args.num_envs)]
    )

    # Load pretrained policy
    if args.algo == "ppo":
        agent = Agent(envs)
    else:
        raise NotImplementedError()

    # Setup model
    agent_path = get_project_folder() / \
        f"runs/train/{args.algo}__{args.env_id}/version_{args.agent_version}"

    state = fabric.load(agent_path / "state.ckpt")
    agent.load_state_dict(state["model"])
    agent.target_layer = args.target_layer
    agent.to(fabric.device)

    # Gather data
    train_b, val_b, test_b = gather_data(
        args, agent_path / "data.joblib", agent, fabric.device)

    return info | {"agent": agent, "fabric": fabric, "args": args, "envs": envs,
                   "train_b": train_b, "val_b": val_b, "test_b": test_b, "mode": args.mode}


def main():
    info = setup()
    training_loader, val_loader, _ = setup_dataloader(info)
    if info["args"].load_version < 0:
        info["reducer"], info["kmeans_expert"] = learn_concepts(info["mode"], info["fabric"].setup(info["agent"]),
                                                                training_loader, info["args"].n_concepts,
                                                                info["args"].n_experts, info["args"].seed)

        info["clf"] = train(info, training_loader, val_loader)

    return info


if __name__ == "__main__":
    GLOBAL_INFO = main()
