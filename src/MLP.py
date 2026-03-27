import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        device,
        layers,
        seed=None,
    ):
        super().__init__()
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)

        self.layers = list(layers)
        self.hidden = nn.ModuleList()
        self.activation = nn.Tanh()

        input_layer = nn.Linear(self.layers[0], self.layers[1], bias=True)
        nn.init.xavier_normal_(input_layer.weight.data, gain=1.0)
        nn.init.zeros_(input_layer.bias.data)
        self.hidden.append(input_layer)

        for input_size, output_size in zip(self.layers[1:-1], self.layers[2:-1]):
            linear = nn.Linear(input_size, output_size, bias=True)
            nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            nn.init.zeros_(linear.bias.data)
            self.hidden.append(linear)

        output_layer = nn.Linear(self.layers[-2], self.layers[-1], bias=False)
        nn.init.xavier_normal_(output_layer.weight.data, gain=1.0)
        self.hidden.append(output_layer)

    def forward(self, input_tensor):
        x = input_tensor
        for index, linear_transform in enumerate(self.hidden):
            if index < len(self.hidden) - 1:
                x = self.activation(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

    def network_parameters(self):
        return list(self.parameters())
