"""Multilayer perceptron, used as the observation-only baseline and as the
backbone of the PINN.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected network with tanh activations.

    Weights use Xavier-normal initialization, hidden biases start at zero,
    and the output layer has no bias.

    Args:
        device: torch device stored on the model (used by the losses).
        layer_sizes: list of layer widths, e.g. ``[1, 40, 1]`` for one
            input, one hidden layer of 40 units and one output.
        seed: if given, seeds ``torch`` before initializing the weights.
    """

    def __init__(self, device, layer_sizes, seed=None):
        super().__init__()
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        self.layer_sizes = list(layer_sizes)
        self.linear_layers = nn.ModuleList()
        self.activation = nn.Tanh()

        input_layer = nn.Linear(self.layer_sizes[0], self.layer_sizes[1], bias=True)
        nn.init.xavier_normal_(input_layer.weight.data, gain=1.0)
        nn.init.zeros_(input_layer.bias.data)
        self.linear_layers.append(input_layer)

        for input_size, output_size in zip(
            self.layer_sizes[1:-1],
            self.layer_sizes[2:-1],
        ):
            hidden_layer = nn.Linear(input_size, output_size, bias=True)
            nn.init.xavier_normal_(hidden_layer.weight.data, gain=1.0)
            nn.init.zeros_(hidden_layer.bias.data)
            self.linear_layers.append(hidden_layer)

        output_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=False,
        )
        nn.init.xavier_normal_(output_layer.weight.data, gain=1.0)
        self.linear_layers.append(output_layer)

    def forward(self, input_tensor):
        """Evaluate the network pointwise.

        Args:
            input_tensor: tensor of shape ``[N, layer_sizes[0]]``.

        Returns:
            Tensor of shape ``[N, layer_sizes[-1]]``.
        """
        x = input_tensor

        for layer_index, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if layer_index < len(self.linear_layers) - 1:
                x = self.activation(x)

        return x

    def network_parameters(self):
        """Return the trainable weights of the network (all parameters)."""
        return list(self.parameters())
