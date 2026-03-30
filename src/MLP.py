import torch
import torch.nn as nn


class MLP(nn.Module):
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
        x = input_tensor

        for layer_index, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if layer_index < len(self.linear_layers) - 1:
                x = self.activation(x)

        return x

    def network_parameters(self):
        return list(self.parameters())
