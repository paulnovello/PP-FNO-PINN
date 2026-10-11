"""Physics-Informed Neural Network: an MLP that approximates ``h(x)`` and also
carries the Strickler control values ``k`` as an optionally trainable
parameter.
"""

import torch
import torch.nn as nn

from MLP import MLP


class PINN(MLP):
    """MLP surrogate for ``h(x)`` together with the ``K_s`` control values.

    ``k`` is stored divided by ``k_ref`` so the optimizer works on values
    of order one.

    Args:
        device: torch device.
        layer_sizes: layer widths of the underlying MLP, e.g. ``[1, 40, 1]``.
        initial_k: initial control values of ``K_s`` (tensor of size ``p``).
        k_ref: scaling factor; the stored parameter is ``initial_k / k_ref``.
        seed: if given, seeds ``torch`` before initializing the weights.
        trainable_k: whether ``k`` receives gradients (inverse problem) or
            stays fixed (forward problem).
    """

    def __init__(
        self,
        device,
        layer_sizes,
        initial_k,
        k_ref=1.0,
        seed=None,
        trainable_k=False,
    ):
        super().__init__(device=device, layer_sizes=layer_sizes, seed=seed)

        self.k_ref = float(k_ref)
        parameter_tensor = torch.as_tensor(
            initial_k,
            device=device,
            dtype=torch.float32,
        ).detach().clone()
        self.k = nn.Parameter(
            parameter_tensor / self.k_ref,
            requires_grad=trainable_k,
        )
        self.dim_k = int(self.k.numel())

    def parameter_values(self):
        """Return the physical control values of ``K_s`` (``k_ref * k``)."""
        return self.k_ref * self.k

    def clamp_parameters(self, min_value=10.0, max_value=100.0):
        """Clamp the physical ``K_s`` values in place to ``[min_value, max_value]``."""
        with torch.no_grad():
            self.k.clamp_(
                min=min_value / self.k_ref,
                max=max_value / self.k_ref,
            )

    def network_parameters(self):
        """Return the network weights only, excluding ``k``."""
        return [
            parameter
            for name, parameter in self.named_parameters()
            if name != "k"
        ]
