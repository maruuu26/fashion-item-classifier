# src/train/train_classifier.py
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data.fashion_mnist import get_dataloaders
from src.models.baseline import build_resnet18


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute batch accuracy from raw logits."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    log_every: int = 100,
    max_batches: int | None = None,  # for quick smoke runs
):
    """
    Standard train loop for one epoch with optional step-level logging.
    """
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    for step, (xb, yb) in enumerate(dataloader, start=1):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)                # forward
        loss = loss_fn(logits, yb)        # scalar
        loss.backward()                   # grads
        optimizer.step()                  # update

        acc = accuracy_from_logits(logits, yb).item()
        running_loss += loss.item()
        running_acc  += acc
        n_batches    += 1

        if step % log_every == 0:
            global_step = (epoch - 1) * len(dataloader) + step
            print(f"  step {step}: loss={loss.item():.4f} acc={acc:.3f}")
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            writer.add_scalar("acc/train_step",  acc,        global_step)

        if max_batches is not None and step >= max_batches:
            break

    epoch_loss = running_loss / max(1, n_batches)
    epoch_acc  = running_acc  / max(1, n_batches)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
):
    """Eval loop with no grad + epoch-level metrics."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        acc  = accuracy_from_logits(logits, yb).item()

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def save_checkpoint(model: torch.nn.Module, path: str = "models/checkpoints/best.pt"):
    """Save model weights only (state_dict)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    # ----- config (tweak as you like) -----
    batch_size   = 64
    num_workers  = 0          # start 0 on macOS; bump to 2 if stable
    img_size     = 224
    epochs       = 1          # start with 1 while fast_dev_run=True
    lr           = 1e-3
    fast_dev_run = True       # set False for full training
    log_every    = 10         # more frequent logs during dev
    max_batches  = 50 if fast_dev_run else None

    run_name   = f"m1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ckpt_path  = "models/checkpoints/best.pt"

    # ----- device -----
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    # ----- data -----
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, img_size=img_size
    )

    # ----- model / loss / optimizer -----
    model = build_resnet18(num_classes=10, pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- logging -----
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer,
            log_every=log_every, max_batches=max_batches
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        # epoch-level logs
        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        writer.add_scalar("acc/train_epoch",  train_acc,  epoch)
        writer.add_scalar("loss/val_epoch",   val_loss,   epoch)
        writer.add_scalar("acc/val_epoch",    val_acc,    epoch)

        print(
            f"[epoch {epoch}] "
            f"train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, ckpt_path)
            print(f"  ↳ new best val_acc={best_val_acc:.3f} — saved {ckpt_path}")

    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
