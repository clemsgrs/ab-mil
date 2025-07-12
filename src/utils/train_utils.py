import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections.abc import Callable

from src.metrics import get_metrics
from src.components import NLLSurvLoss


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):

            idx, x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, attn = model(x)
            loss = criterion(logits, label)

            loss_value = loss.item()
            epoch_loss += loss_value

            if gradient_accumulation:
                loss = loss / gradient_accumulation

            loss.backward()

            if not gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
            elif (i + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            pred = torch.topk(logits, 1, dim=1)[1]
            preds.extend(pred[:, 0].clone().tolist())

            prob = F.softmax(logits, dim=1).cpu().detach().numpy()
            probs = np.append(probs, prob, axis=0)

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Tune"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for batch in t:

                idx, x, label = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits, attn = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def inference(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    metric_names: list[str],
    batch_size: int = 1,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Inference"),
        unit=" case",
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for batch in t:
                idx, x, label = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits, attn = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    return results


class LossFactory:
    def __init__(
        self,
        task: str,
    ):
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        elif task == "survival":
            self.criterion = NLLSurvLoss()

    def get_loss(self):
        return self.criterion


class OptimizerFactory:
    def __init__(
        self,
        name: str,
        params: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        if name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise KeyError(f"{name} not supported")

    def get_optimizer(self):
        return self.optimizer


class SchedulerFactory:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: dict | None = None,
    ):
        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            assert (
                params.T_max != -1
            ), "T_max parameter must be specified! If you dont know what to use, plug in nepochs"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                params.T_max, eta_min=params.eta_min
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.mode,
                factor=params.factor,
                patience=params.patience,
                min_lr=params.min_lr,
            )
        elif self.name:
            raise KeyError(f"{self.name} not supported")

    def get_scheduler(self):
        return self.scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Path | None = None,
        save_all: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_all = save_all
        self.verbose = verbose

        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, model, results):
        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            fname = f"best.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch + 1 and self.verbose:
                print(
                    f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                )
            elif self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f"epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(model.state_dict(), Path(self.checkpoint_dir, "latest.pt"))
