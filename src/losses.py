from dataclasses import dataclass

import torch

from Backwater_model import Ks_function, bathymetry_interpolator, g, h_BC, q, regime
from normalization import input_scale, normalize_input


@dataclass
class LossBreakdown:
    total: torch.Tensor
    residual: torch.Tensor
    observation: torch.Tensor
    boundary: torch.Tensor


@dataclass(frozen=True)
class LossWeights:
    residual: float
    observation: float
    boundary: float


@dataclass
class LossNormalizationState:
    residual_scale: torch.Tensor | None = None
    observation_scale: torch.Tensor | None = None
    boundary_scale: torch.Tensor | None = None


def observation_loss(model, obs, col):
    if obs.shape[0] == 0:
        return torch.tensor(0.0, device=model.device)

    predictions = model(normalize_input(obs[:, :1], col))
    observations = obs[:, 1:2]
    return torch.norm(predictions - observations, p=2) ** 2 / obs.shape[0]


def boundary_condition_loss(h_boundary):
    return torch.norm(h_boundary - h_BC, p=2) ** 2


def residual_loss(model, col):
    domain = col
    if not domain.requires_grad:
        domain = domain.detach().clone().requires_grad_(True)

    n_col = domain.shape[0]
    parameter_function = Ks_function(domain, model.parameter_values(), col)
    b_prime = bathymetry_interpolator(model.device, domain)[1]
    normalized_domain = (
        normalize_input(domain, col).detach().clone().requires_grad_(True)
    )

    h = model(normalized_domain).clamp(min=1e-3)
    h_x = torch.autograd.grad(
        h,
        normalized_domain,
        grad_outputs=torch.ones_like(h),
        create_graph=True,
        retain_graph=True,
    )[0] / input_scale(col)

    froude = q / (g * h**3) ** (1 / 2)
    friction = q**2 / (parameter_function**2 * h ** (10 / 3))
    residual = h_x + (b_prime + friction) / (1 - froude**2)
    return torch.norm(residual, p=2) ** 2 / n_col


def _safe_scale(loss_value):
    scale = loss_value.detach().clone()
    return torch.clamp(scale, min=1e-12)


def physics_informed_losses(model, col, obs, *, weights, normalize, state):
    if regime != "subcritical":
        raise NotImplementedError("Only the subcritical regime is supported.")

    boundary_value = model(normalize_input(col, col))[-1]
    residual = residual_loss(model, col)
    observation = observation_loss(model, obs, col)
    boundary = boundary_condition_loss(boundary_value)

    if normalize and state is not None:
        if state.residual_scale is None:
            state.residual_scale = _safe_scale(residual)
        if state.observation_scale is None:
            state.observation_scale = _safe_scale(observation)
        if state.boundary_scale is None:
            state.boundary_scale = _safe_scale(boundary)

        residual = residual / state.residual_scale
        observation = observation / state.observation_scale
        boundary = boundary / state.boundary_scale

    residual = weights.residual * residual
    observation = weights.observation * observation
    boundary = weights.boundary * boundary
    total = residual + observation + boundary

    return LossBreakdown(
        total=total,
        residual=residual,
        observation=observation,
        boundary=boundary,
    )
