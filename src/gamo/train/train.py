import os

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from gamo.model.model import DEFAULT_HIDDEN_SIZES, SimpleMLP
from gamo.utils.data import get_dataloaders
from gamo.utils.environment import get_device, set_seed
from gamo.utils.model_utils import CHECKPOINT_NAME, evaluate
from gamo.utils.paths import checkpoint_path, write_json_atomic

TRAIN_SEED = 1337
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 128
TRAIN_LEARNING_RATE = 1e-3
TRAIN_MIN_LEARNING_RATE = 1e-5
TRAIN_WEIGHT_DECAY = 1e-3
TRAIN_VALIDATION_SPLIT = 0.3


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / max(1, total_samples)


def train(
    num_epochs: int,
    device: torch.device,
    dataloader,
    val_loader,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    model_kwargs: dict | None = None,
    checkpoint_out: str | None = None,
) -> nn.Module:
    """Train a SimpleMLP from scratch and keep the best-validation epoch.

    Recipe: AdamW · cosine LR to 1e-5 · 10 epochs · batch 128.

    Regularisation is weight decay, a plain MLP starts to overfit after a few epochs, so we train for only
    a few epochs and keep the best-validation snapshot. Expect ~88% test accuracy

    Training history (losses and learning rate per epoch) is saved next to the checkpoint.
    """
    model_kwargs = model_kwargs or {}
    model = SimpleMLP(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=TRAIN_MIN_LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    # --- Training loop (keep the best-validation snapshot) ---
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, dataloader, optimizer, device, criterion)
        validation = evaluate(model, val_loader, criterion, device, show_progress=False)
        val_loss = validation["loss"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # --- Restore and save the best epoch ---
    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_out is not None:
            os.makedirs(os.path.dirname(checkpoint_out) or ".", exist_ok=True)
            save_file(best_state, checkpoint_out)
            # Save training history alongside checkpoint
            history_out = os.path.splitext(checkpoint_out)[0] + "_history.json"
            write_json_atomic(history_out, history)
            print(
                f"Best model saved to {checkpoint_out} (val loss {best_val_loss:.4f})"
            )
            print(f"Training history saved to {history_out}")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model


def train_model(force: bool = False) -> nn.Module | None:
    """Train the one checkpoint used by all final experiments, unless it already exists."""
    output = checkpoint_path(CHECKPOINT_NAME, mkdir=True)
    if os.path.exists(output) and not force:
        print(f"Checkpoint already exists at {output}; skipping training")
        return None

    set_seed(TRAIN_SEED)
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=TRAIN_BATCH_SIZE,
        val_split=TRAIN_VALIDATION_SPLIT,
        seed=TRAIN_SEED,
    )
    return train(
        num_epochs=TRAIN_EPOCHS,
        device=device,
        dataloader=train_loader,
        val_loader=val_loader,
        learning_rate=TRAIN_LEARNING_RATE,
        weight_decay=TRAIN_WEIGHT_DECAY,
        model_kwargs={"num_classes": 10, "hidden_sizes": DEFAULT_HIDDEN_SIZES},
        checkpoint_out=output,
    )
