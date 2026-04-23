"""Generic train/eval loop with early stopping, scheduler and W&B logging."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import copy

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .device import supports_amp


@dataclass
class TrainConfig:
    """Hyperparameter container shared with W&B."""

    experiment: str
    model_name: str
    num_classes: int
    epochs: int = 8
    batch_size: int = 16
    image_size: int = 224
    learning_rate: float = 1e-3
    backbone_lr_factor: float = 0.1
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    label_smoothing: float = 0.1
    dropout: float = 0.2
    early_stopping_patience: int = 4
    scheduler: str = "plateau"  # plateau | cosine
    scheduler_patience: int = 2
    scheduler_factor: float = 0.3
    cosine_warmup_epochs: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999
    seed: int = 42
    transfer_strategy: str = "fine_tuning"  # feature_extraction | partial | fine_tuning
    extra: dict = field(default_factory=dict)


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    ema_model: AveragedModel | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = scaler is not None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _update_bn(ema_model: AveragedModel, loader: DataLoader, device: torch.device) -> None:
    """Refresh BN running stats on the EMA model — no-op if no BN modules."""
    bn_modules = [m for m in ema_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return
    was_training = ema_model.training
    ema_model.train()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        ema_model(x)
    ema_model.train(was_training)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    output_dir: Path,
    param_groups=None,
    use_wandb: bool = True,
):
    """Train ``model`` and return ``(best_val_acc, best_path, history)``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optim_params = param_groups if param_groups is not None else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.learning_rate}]
    optimizer = make_optimizer(cfg.optimizer, optim_params, cfg.learning_rate, cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - cfg.cosine_warmup_epochs, 1))
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and supports_amp(device)) else None

    ema_model = None
    if cfg.use_ema:
        decay = cfg.ema_decay
        def _ema_avg(avg_p, new_p, num_avg):
            return decay * avg_p + (1.0 - decay) * new_p
        ema_model = AveragedModel(model, avg_fn=_ema_avg)
        ema_model.to(device)

    history = []
    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"
    epochs_without_improve = 0
    initial_lrs = [g["lr"] for g in optimizer.param_groups]

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Linear warmup for cosine scheduler
        if cfg.scheduler == "cosine" and epoch <= cfg.cosine_warmup_epochs:
            warmup_scale = epoch / max(cfg.cosine_warmup_epochs, 1)
            for g, lr0 in zip(optimizer.param_groups, initial_lrs):
                g["lr"] = lr0 * warmup_scale

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, ema_model=ema_model,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if ema_model is not None:
            # Update BN stats for EMA model (lightweight — full pass over train loader)
            _update_bn(ema_model, train_loader, device)
            ema_val_loss, ema_val_acc = evaluate(ema_model, val_loader, criterion, device)
        else:
            ema_val_loss, ema_val_acc = None, None

        epoch_time = time.time() - t0

        if cfg.scheduler == "cosine" and epoch > cfg.cosine_warmup_epochs:
            scheduler.step()
        elif cfg.scheduler == "plateau":
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }
        if ema_val_acc is not None:
            log_payload["ema_val_accuracy"] = ema_val_acc
            log_payload["ema_val_loss"] = ema_val_loss
        history.append(log_payload)
        if use_wandb and wandb.run is not None:
            wandb.log(log_payload)

        # Track best by either raw or EMA val acc (pick whichever is higher)
        candidate_acc = max(val_acc, ema_val_acc) if ema_val_acc is not None else val_acc
        use_ema_weights = ema_val_acc is not None and ema_val_acc > val_acc
        improved = candidate_acc > best_val_acc
        if improved:
            best_val_acc = candidate_acc
            state_dict_to_save = (
                ema_model.module.state_dict() if use_ema_weights else model.state_dict()
            )
            torch.save(
                {
                    "model_state_dict": state_dict_to_save,
                    "config": asdict(cfg),
                    "val_accuracy": candidate_acc,
                    "epoch": epoch,
                    "ema_used": use_ema_weights,
                },
                best_path,
            )
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        ema_tag = f" - ema_val {ema_val_acc:.3f}" if ema_val_acc is not None else ""
        print(
            f"[{cfg.experiment}] epoch {epoch:02d}/{cfg.epochs} - "
            f"train_loss {train_loss:.3f} acc {train_acc:.3f} - "
            f"val_loss {val_loss:.3f} acc {val_acc:.3f}{ema_tag} - "
            f"lr {current_lr:.1e} - {epoch_time:.1f}s",
            flush=True,
        )

        if epochs_without_improve >= cfg.early_stopping_patience:
            print(f"[{cfg.experiment}] early stopping at epoch {epoch}", flush=True)
            break

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_val_acc, best_path, history
