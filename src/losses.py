import torch

from Backwater_model import Ks_function, bathymetry_interpolator, g, h_BC, q, regime
from normalization import input_scale, normalize_input


def observation_loss(model, observation_points, collocation_points):
    if observation_points.shape[0] == 0:
        return torch.tensor(0.0, device=model.device)

    # Step 1: normalize the observation coordinates before using the model.
    observation_coordinates = observation_points[:, :1]
    normalized_observation_coordinates = normalize_input(
        observation_coordinates,
        collocation_points,
    )

    # Step 2: compare the model prediction to the observed water height.
    predicted_observations = model(normalized_observation_coordinates)
    true_observations = observation_points[:, 1:2]
    observation_error = predicted_observations - true_observations

    return torch.norm(observation_error, p=2) ** 2 / observation_points.shape[0]


def boundary_condition_loss(boundary_value):
    boundary_error = boundary_value - h_BC
    return torch.norm(boundary_error, p=2) ** 2


def residual_loss(model, collocation_points):
    x = collocation_points
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    number_of_points = x.shape[0]

    # Step 1: evaluate the physical ingredients of the PDE.
    parameter_function = Ks_function(x, model.parameter_values(), collocation_points)
    bathymetry_slope = bathymetry_interpolator(x)[1].to(model.device)

    # Step 2: predict the water height on normalized coordinates.
    normalized_x = normalize_input(x, collocation_points).detach().clone()
    normalized_x.requires_grad_(True)


    # TODO: compute predicted_height. Clam its value using .clamp(min=1e-3)

    # Step 3: compute the derivatives.
    # TODO: compute height_gradient using torch.autograd. Use grad_outputs=torch.ones_like(something). What value should you use for rcreate_graph and retain_graph ? At the end, add a [0].

    x_scale = input_scale(collocation_points)
    height_gradient = height_gradient / x_scale

    # Step 4: build the backwater residual.
    # TODO: build residual using the Back Water equation

    return torch.norm(residual, p=2) ** 2 / number_of_points


def _safe_scale(loss_value):
    copied_loss = loss_value.detach().clone()
    return torch.clamp(copied_loss, min=1e-12)


def physics_informed_loss(
    model,
    collocation_points,
    observation_points,
    *,
    lambda_residual,
    lambda_observation,
    lambda_boundary,
    scale_losses,
    loss_scales,
):
    if regime != "subcritical":
        raise NotImplementedError("Only the subcritical regime is supported.")

    # Step 1: compute the three raw physical losses.
    normalized_collocation_points = normalize_input(
        collocation_points,
        collocation_points,
    )
    boundary_value = model(normalized_collocation_points)[-1]
    residual_loss_value = residual_loss(model, collocation_points)
    observation_loss_value = observation_loss(
        model,
        observation_points,
        collocation_points,
    )
    boundary_loss_value = boundary_condition_loss(boundary_value)

    # Step 2: optionally balance the loss magnitudes.
    if scale_losses and loss_scales is not None:
        if loss_scales["residual"] is None:
            loss_scales["residual"] = _safe_scale(residual_loss_value)
        if loss_scales["observation"] is None:
            loss_scales["observation"] = _safe_scale(observation_loss_value)
        if loss_scales["boundary"] is None:
            loss_scales["boundary"] = _safe_scale(boundary_loss_value)

        residual_loss_value = residual_loss_value / loss_scales["residual"]
        observation_loss_value = observation_loss_value / loss_scales["observation"]
        boundary_loss_value = boundary_loss_value / loss_scales["boundary"]

    # Step 3: apply the phase weights and add everything together.
    # TODO: Assemble the total loss using lambdas

    return {
        "total_loss": total_loss,
        "residual_loss": residual_loss_value,
        "observation_loss": observation_loss_value,
        "boundary_loss": boundary_loss_value,
        "loss_scales": loss_scales,
    }
