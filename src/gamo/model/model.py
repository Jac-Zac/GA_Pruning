from collections.abc import Sequence

import torch
import torch.nn as nn

# Single architecture for the whole project (small enough for a simple experiment).
# But still large enough that pruning choices are non-trivial.
DEFAULT_HIDDEN_SIZES = (256, 256, 256)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        num_classes: int = 10,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.num_classes = num_classes
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        prev_size = input_size
        for h_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        *,
        weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run a dense pass or evaluate a population of masks or weights.

        When ``masks`` has shape ``(population, n_hidden_neurons)``, the input
        batch is shared across the population and each mask is applied after
        every hidden ReLU. The returned logits then have shape ``(population, batch, num_classes)``.

        ``weights`` may instead provide population-batched weight and bias
        tensors in parameter order.
        """
        # NOTE: One is used for structured mask evaluation, the other for unstructured mask evaluation.
        if masks is not None and weights is not None:
            raise ValueError("masks and weights cannot be passed together")

        x = self.flatten(x)
        if weights is not None:
            return self._forward_with_weights(x, weights)
        if masks is None:
            return self.output_layer(self.hidden_layers(x))

        expected = sum(self.hidden_sizes)
        if masks.ndim != 2 or masks.shape[1] != expected:
            raise ValueError(f"masks must have shape (population, {expected})")

        pop = masks.shape[0]
        h = x.unsqueeze(0).expand(pop, -1, -1)
        offset = 0
        for index, width in enumerate(self.hidden_sizes):
            linear = self.hidden_layers[2 * index]
            relu = self.hidden_layers[2 * index + 1]
            h = relu(linear(h))
            # Each hidden layer consumes its own slice of the flat neuron mask.
            layer_mask = masks[:, offset : offset + width].unsqueeze(1)
            h = h * layer_mask
            offset += width
        return self.output_layer(h)

    def _forward_with_weights(
        self, x: torch.Tensor, weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate population-batched parameters on a shared input batch."""
        expected = 2 * (len(self.hidden_sizes) + 1)
        if len(weights) != expected:
            raise ValueError(f"expected {expected} weight and bias tensors")

        pop = weights[0].shape[0]
        h = x.unsqueeze(0).expand(pop, -1, -1)

        # Parameters alternate weight and bias; baddbmm computes h @ W.T + b.
        hidden_weights = weights[:-2:2]
        hidden_biases = weights[1:-2:2]
        for weight, bias in zip(hidden_weights, hidden_biases, strict=True):
            h = torch.relu(torch.baddbmm(bias.unsqueeze(1), h, weight.transpose(1, 2)))

        output_weight, output_bias = weights[-2:]
        return torch.baddbmm(output_bias.unsqueeze(1), h, output_weight.transpose(1, 2))
