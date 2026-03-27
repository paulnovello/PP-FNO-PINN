import torch


def normalize_input(input_tensor, reference_grid):
    reference_min = torch.min(reference_grid)
    reference_max = torch.max(reference_grid)
    return (input_tensor - reference_min) / (reference_max - reference_min)


def input_scale(reference_grid):
    return torch.max(reference_grid) - torch.min(reference_grid)
