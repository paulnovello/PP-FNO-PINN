import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _configure_runtime_environment():
    cache_root = Path("/tmp/pp_fno_pinn_cache")
    matplotlib_root = cache_root / "matplotlib"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))


_configure_runtime_environment()

from Backwater_model import Ks_function, compute_ref_solution
from FNO import FNO
from MLP import MLP
from normalization import normalize_input
from PINN import PINN
from trainer import PITrainer, Trainer


@dataclass(frozen=True)
class LearningPdeMetrics:
    rmse: float


@dataclass(frozen=True)
class InverseProblemMetrics:
    solution_rmse: float
    parameter_error: float


def default_device(use_gpu=True):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)
    return device


def default_problem(device, seed=0):
    del seed
    k_true = torch.tensor([45, 38, 28, 22, 30, 45, 60, 75], device=device)
    collocation_points = torch.linspace(0, 1000, 100, device=device).view(-1, 1)
    collocation_points.requires_grad_(True)
    return k_true, collocation_points


def build_observations(ref_solution, n_obs, seed=None, noise_std=0, grid=True):
    if seed is not None:
        torch.manual_seed(seed)

    device = ref_solution["domain"].device
    n_domain = ref_solution["domain"].shape[0]
    if n_obs == 0:
        return torch.empty((0, 2), device=device)

    if grid:
        indices = torch.linspace(0, n_domain - 1, steps=n_obs, device=device).long()
    else:
        indices = torch.randperm(n_domain, device=device)[:n_obs]

    noisy_solution = ref_solution["solution"].detach().clone() + torch.randn(
        n_domain,
        1,
        device=device,
    ) * noise_std
    return torch.hstack((ref_solution["domain"], noisy_solution))[indices].detach().clone()


def build_pinn(device, k_true, seed=0, k_ref=1.0, trainable_k=False):
    return PINN(
        device=device,
        layer_sizes=[1, 40, 1],
        initial_k=k_true,
        k_ref=k_ref,
        seed=seed,
        trainable_k=trainable_k,
    )


def build_mlp(device, seed=0):
    return MLP(
        device=device,
        layer_sizes=[1, 40, 1],
        seed=seed,
    )


def build_fno(device, seed=0, n_modes_layer_1=20, n_modes_layer_2=20, n_channels=10):
    return FNO(
        device=device,
        n_modes_layer_1=n_modes_layer_1,
        n_modes_layer_2=n_modes_layer_2,
        n_channels=n_channels,
        seed=seed,
    )


def relative_l2_error(estimate, truth):
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    return float(
        np.linalg.norm(estimate_array - truth_array, ord=2)
        / np.linalg.norm(truth_array, ord=2)
    )


def _build_reference_problem(seed=0, use_gpu=False):
    device = default_device(use_gpu=use_gpu)
    k_true, collocation_points = default_problem(device, seed=seed)
    reference_model = build_pinn(device, k_true, seed=seed)
    ref_solution = compute_ref_solution(
        collocation_points,
        k=k_true,
        dx=10,
    )
    return device, k_true, collocation_points, ref_solution


def run_learning_pde_solution_case(
    model_kind,
    seed=0,
    n_obs=40,
    use_gpu=False,
):
    device, k_true, collocation_points, ref_solution = _build_reference_problem(
        seed=seed,
        use_gpu=use_gpu,
    )
    observation_points = build_observations(
        ref_solution,
        n_obs=n_obs,
        seed=None,
        grid=True,
    )

    if model_kind == "fno":
        model = build_fno(device, seed=seed)
        Trainer().fit(
            model,
            collocation_points,
            observation_points,
            ref_solution,
            n_iter=300,
            lr=1e-2,
            display_freq=(1000, 1000),
        )
    elif model_kind == "mlp":
        model = build_mlp(device, seed=seed)
        Trainer(optimizer_name="lbfgs").fit(
            model,
            collocation_points,
            observation_points,
            ref_solution,
            n_iter=300,
            lr=1e-2,
            display_freq=(1000, 1000),
        )
    elif model_kind == "pinn":
        model = build_pinn(device, k_true, seed=seed)
        PITrainer(train_k=False).fit(
            model,
            collocation_points,
            observation_points,
            ref_solution,
            pre_train_iter=100,
            alter_steps=4,
            alter_freq=(40, 0),
            scale_losses=True,
            display_freq=(1000, 1000),
        )
    else:
        raise ValueError("model_kind must be 'fno', 'mlp', or 'pinn'")

    h_true = ref_solution["solution"].detach().clone().cpu().numpy().flatten()
    h_pred = (
        model(normalize_input(collocation_points, collocation_points))
        .detach()
        .clone()
        .cpu()
        .numpy()
        .flatten()
    )
    return LearningPdeMetrics(rmse=relative_l2_error(h_pred, h_true))


def build_inverse_problem_model(device, k_true, seed=0):
    return build_pinn(
        device,
        40 * torch.ones_like(k_true),
        seed=seed,
        k_ref=40,
        trainable_k=True,
    )


def run_inverse_problem_case(seed=0, n_obs=40, use_gpu=False):
    device, k_true, collocation_points, ref_solution = _build_reference_problem(
        seed=seed,
        use_gpu=use_gpu,
    )
    observation_points = build_observations(
        ref_solution,
        n_obs=n_obs,
        seed=seed,
        grid=False,
    )
    model = build_inverse_problem_model(device, k_true, seed=seed)

    PITrainer(train_k=True).fit(
        model,
        collocation_points,
        observation_points,
        ref_solution,
        pre_train_iter=100,
        alter_steps=4,
        alter_freq=(40, 10),
        scale_losses=True,
        display_freq=(1000, 1000),
    )

    h_true = ref_solution["solution"].detach().clone().cpu().numpy().flatten()
    h_pred = (
        model(normalize_input(ref_solution["domain"], collocation_points))
        .detach()
        .clone()
        .cpu()
        .numpy()
        .flatten()
    )
    ks_true = ref_solution["parameter_function"].detach().clone().cpu().numpy().flatten()
    ks_pred = (
        Ks_function(
            ref_solution["domain"],
            model.parameter_values(),
            collocation_points,
        )
        .detach()
        .clone()
        .cpu()
        .numpy()
        .flatten()
    )

    return InverseProblemMetrics(
        solution_rmse=relative_l2_error(h_pred, h_true),
        parameter_error=relative_l2_error(ks_pred, ks_true),
    )
