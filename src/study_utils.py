import matplotlib.pyplot as plt
import numpy as np
import torch

from Backwater_model import compute_ref_solution
from FNO import FNO
from MLP import MLP
from normalization import normalize_input
from PINN import PINN
from trainer import PITrainer, Trainer


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
    col = torch.linspace(0, 1000, 100, device=device).view(-1, 1)
    col.requires_grad_(True)
    return k_true, col


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
        layers=[1, 40, 1],
        k_0=k_true,
        k_ref=k_ref,
        seed=seed,
        trainable_k=trainable_k,
    )


def build_mlp(device, seed=0):
    return MLP(
        device=device,
        layers=[1, 40, 1],
        seed=seed,
    )


def build_fno(device, seed=0, layers=(20, 20), n_enc=10):
    return FNO(
        device=device,
        layers=list(layers),
        n_enc=n_enc,
        seed=seed,
    )


def relative_l2_error(estimate, truth):
    return float(
        np.linalg.norm(estimate - truth, ord=2) / np.linalg.norm(truth, ord=2)
    )


def run_n_obs_study(
    model_kind,
    use_pinn=True,
    n_obs_values=(10, 20, 40, 60, 80, 100),
    seed=0,
    use_gpu=True,
):
    device = default_device(use_gpu=use_gpu)
    k_true, col = default_problem(device, seed=seed)
    reference_model = build_pinn(device, k_true, seed=seed)
    ref_solution = compute_ref_solution(reference_model, col, k=k_true, dx=10)

    results = []
    for n_obs in n_obs_values:
        obs = build_observations(ref_solution, n_obs=n_obs, seed=None, grid=True)

        if model_kind == "fno":
            model = build_fno(device, seed=seed)
            trainer = Trainer()
            trainer.fit(
                model,
                col,
                obs,
                ref_solution,
                n_iter=300,
                lr=1e-2,
                display_freq=(500, 500),
            )
        elif model_kind == "pinn" and use_pinn:
            model = build_pinn(device, k_true, seed=seed)
            trainer = PITrainer(train_k=False)
            trainer.fit(
                model,
                col,
                obs,
                ref_solution,
                pre_train_iter=100,
                alter_steps=4,
                alter_freq=(40, 0),
                normalize_losses=True,
                display_freq=(500, 500),
            )
        elif model_kind == "pinn":
            model = build_mlp(device, seed=seed)
            trainer = Trainer(optimizer_name="lbfgs")
            trainer.fit(
                model,
                col,
                obs,
                ref_solution,
                n_iter=300,
                lr=1e-2,
                display_freq=(500, 500),
            )
        else:
            raise ValueError("model_kind must be 'fno' or 'pinn'")

        h_true = ref_solution["solution"].detach().clone().cpu().numpy().flatten()
        h_pred = model(normalize_input(col, col)).detach().clone().cpu().numpy().flatten()
        results.append(relative_l2_error(h_pred, h_true))

    return {
        "model_kind": model_kind,
        "use_pinn": use_pinn,
        "n_obs": np.array(n_obs_values),
        "rmse": np.array(results),
    }


def save_n_obs_results(path, results):
    np.savez(
        path,
        n_obs=results["n_obs"],
        rmse=results["rmse"],
        model_kind=results["model_kind"],
        use_pinn=results["use_pinn"],
    )


def load_n_obs_results(path):
    data = np.load(path, allow_pickle=True)
    return {
        "n_obs": data["n_obs"],
        "rmse": data["rmse"],
        "model_kind": str(data["model_kind"]),
        "use_pinn": bool(data["use_pinn"]),
    }


def plot_n_obs_study(fno_results, mlp_results, pinn_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fno_results["n_obs"], fno_results["rmse"], "o-", label="FNO")
    ax.plot(mlp_results["n_obs"], mlp_results["rmse"], "s-", label="MLP without PINN")
    ax.plot(pinn_results["n_obs"], pinn_results["rmse"], "^-", label="MLP with PINN")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Relative RMSE")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    ax.set_title("Effect of N_obs on the solution-learning error")
    return fig, ax
