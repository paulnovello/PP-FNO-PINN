import os
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pp_fno_pinn_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pp_fno_pinn_cache/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Backwater_model import compute_ref_solution
from FNO import FNO
from MLP import MLP
from experiment_runner import run_inverse_problem_case, run_learning_pde_solution_case
from losses import physics_informed_loss
from normalization import normalize_input
from PINN import PINN


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


def build_fno(device, seed=0, n_modes_layer_1=20, n_modes_layer_2=20, n_channels=10):
    return FNO(
        device=device,
        n_modes_layer_1=n_modes_layer_1,
        n_modes_layer_2=n_modes_layer_2,
        n_channels=n_channels,
        seed=seed,
    )


class NotebookRegressionTests(unittest.TestCase):
    def test_learning_pde_solution_fno_rmse(self):
        metrics = run_learning_pde_solution_case(
            "fno",
            seed=0,
            n_obs=40,
            use_gpu=False,
        )
        self.assertLess(metrics.rmse, 5e-2)

    def test_learning_pde_solution_pinn_rmse(self):
        metrics = run_learning_pde_solution_case(
            "pinn",
            seed=0,
            n_obs=40,
            use_gpu=False,
        )
        self.assertLess(metrics.rmse, 2e-2)

    def test_inverse_problem_pinn_parameter_error(self):
        metrics = run_inverse_problem_case(seed=0, n_obs=40, use_gpu=False)
        self.assertLess(metrics.parameter_error, 1e-1)

    def test_fno_forward_accepts_notebook_input_shape(self):
        device = default_device(use_gpu=False)
        _, collocation_points = default_problem(device, seed=0)
        model = build_fno(device, seed=0)

        normalized_coordinates = normalize_input(
            collocation_points,
            collocation_points,
        )
        predictions = model(normalized_coordinates)

        self.assertEqual(predictions.shape, collocation_points.shape)

    def test_physics_informed_loss_returns_expected_keys(self):
        device = default_device(use_gpu=False)
        k_true, collocation_points = default_problem(device, seed=0)
        model = build_pinn(device, k_true, seed=0)
        ref_solution = compute_ref_solution(
            model,
            collocation_points,
            k=k_true,
            dx=10,
        )
        observation_points = build_observations(
            ref_solution,
            n_obs=5,
            seed=0,
            grid=True,
        )
        loss_scales = {
            "residual": None,
            "observation": None,
            "boundary": None,
        }

        loss_values = physics_informed_loss(
            model,
            collocation_points,
            observation_points,
            residual_weight=1.0,
            observation_weight=1.0,
            boundary_weight=1.0,
            scale_losses=True,
            loss_scales=loss_scales,
        )

        self.assertEqual(
            set(loss_values.keys()),
            {
                "total_loss",
                "residual_loss",
                "observation_loss",
                "boundary_loss",
                "loss_scales",
            },
        )
        summed_terms = (
            loss_values["residual_loss"]
            + loss_values["observation_loss"]
            + loss_values["boundary_loss"]
        )
        self.assertTrue(torch.isclose(loss_values["total_loss"], summed_terms))


if __name__ == "__main__":
    unittest.main()
