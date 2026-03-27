import torch
import torch.nn as nn

from MLP import MLP


class PINN(MLP):
    def __init__(
        self,
        device,
        layers,
        k_0,
        k_ref=1.0,
        seed=None,
        trainable_k=False,
    ):
        super().__init__(
            device=device,
            layers=layers,
            seed=seed,
        )

        self.k_ref = float(k_ref)
        parameter = torch.as_tensor(
            k_0,
            device=device,
            dtype=torch.float32,
        ).detach().clone()
        self.k = nn.Parameter(parameter / self.k_ref, requires_grad=trainable_k)
        self.dim_k = int(self.k.numel())

    def parameter_values(self):
        return self.k_ref * self.k

    def clamp_parameters(self, min_value=10.0, max_value=100.0):
        with torch.no_grad():
            self.k.clamp_(min=min_value / self.k_ref, max=max_value / self.k_ref)

    def network_parameters(self):
        return [parameter for name, parameter in self.named_parameters() if name != "k"]
