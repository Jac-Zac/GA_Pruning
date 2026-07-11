"""Model utilities for checkpoint loading and parameter manipulation.

The trained model is a single safetensors file (weights only), saved as
``artifacts/checkpoints/mlp.safetensors``.
"""

import os
from typing import Callable

import torch
import torch.nn as nn
from safetensors.torch import load_file
from tqdm.auto import tqdm

from gamo.model.model import SimpleMLP

# The project trains a single canonical model, saved under this name.
CHECKPOINT_NAME = "mlp.safetensors"


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> dict:
    """Return mean loss and percentage accuracy for a model and data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(
        data_loader,
        desc="Evaluating",
        leave=False,
        disable=not show_progress,
    ):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": 100 * correct / total}


def load_pretrained_model(
    model_path: str,
    device: torch.device,
    **model_kwargs,
) -> SimpleMLP:
    """Load the canonical ``SimpleMLP`` from a safetensors checkpoint."""
    model = SimpleMLP(**model_kwargs).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at '{model_path}'")
    model.load_state_dict(load_file(model_path, device=str(device)))

    return model


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all model parameters into a single 1-D tensor."""
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def _validate_parameter_count(model: nn.Module, count: int) -> None:
    expected = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise ValueError(f"expected {expected} parameters, received {count}")


def set_params_from_tensor(model: nn.Module, params: torch.Tensor) -> None:
    """Write a flat parameter tensor back into model parameter buffers."""
    _validate_parameter_count(model, params.numel())
    parameters = list(model.parameters())
    chunks = params.split([parameter.numel() for parameter in parameters])
    with torch.no_grad():
        for parameter, chunk in zip(parameters, chunks, strict=True):
            parameter.copy_(chunk.reshape_as(parameter))


def unflatten_batched_params(
    batched_params: torch.Tensor, model_template: nn.Module
) -> list[torch.Tensor]:
    """Reshape flat population parameters to tensors matching the model layers."""
    if batched_params.ndim != 2:
        raise ValueError("batched_params must have shape (population, parameters)")
    _validate_parameter_count(model_template, batched_params.shape[1])
    parameters = list(model_template.parameters())
    chunks = batched_params.split(
        [parameter.numel() for parameter in parameters], dim=1
    )
    return [
        chunk.reshape(batched_params.shape[0], *parameter.shape)
        for parameter, chunk in zip(parameters, chunks, strict=True)
    ]


def eval_weights(
    model_class: Callable[..., nn.Module],
    model_kwargs: dict,
    weights: torch.Tensor,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    show_progress: bool = False,
) -> dict:
    """Instantiate a model, load a flat weight tensor into it, and evaluate on loader."""
    model = model_class(**model_kwargs).to(device)
    set_params_from_tensor(model, weights)
    return evaluate(
        model, loader, nn.CrossEntropyLoss(), device, show_progress=show_progress
    )
