"""Map physical coordinates to [0, 1] and back for network inputs.
"""

import torch


def normalize_input(input_tensor, reference_grid):
    """Min-max normalize ``input_tensor`` using the bounds of ``reference_grid``.

    Args:
        input_tensor: coordinates to normalize.
        reference_grid: grid whose min and max define the mapping.

    Returns:
        Tensor with the same shape as ``input_tensor``.
    """
    reference_min = torch.min(reference_grid)
    reference_max = torch.max(reference_grid)
    return (input_tensor - reference_min) / (reference_max - reference_min)


def input_scale(reference_grid):
    """Return ``max - min`` of ``reference_grid``.

    Use it to convert derivatives with respect to normalized coordinates
    back to physical units.
    """
    return torch.max(reference_grid) - torch.min(reference_grid)
