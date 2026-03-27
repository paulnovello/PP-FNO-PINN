import matplotlib.pyplot as plt
import numpy as np
import torch

from Backwater_model import Ks_function, bathymetry_interpolator
from normalization import normalize_input


def _subdomain_indices(ref_solution):
    n_parameters = int(ref_solution["parameter"].numel())
    subdomains = torch.linspace(
        float(torch.min(ref_solution["domain"]).item()),
        float(torch.max(ref_solution["domain"]).item()),
        n_parameters,
        device=ref_solution["domain"].device,
    ).view(-1, 1)
    indices = (
        (subdomains / ref_solution["dx"])
        .clamp(max=ref_solution["domain"].shape[0] - 1)
        .int()
        .detach()
        .clone()
        .cpu()
        .numpy()
    )
    return subdomains, indices


def display_data(model, col, ref_solution, obs):
    del col
    subdomains, indices = _subdomain_indices(ref_solution)

    fig, ax = plt.subplots()
    if obs.shape[0] < 5:
        ax.set_title(
            "Reference solution and data, "
            f"$K_s = {list(np.around(ref_solution['parameter'].detach().clone().cpu().numpy(), 2))}"
            " \\ m^{1/3}.s^{-1}$"
        )
    else:
        ax.set_title("Reference solution and data")

    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["solution"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        color="#1f77b4",
        label="RK4 solution",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["normal height"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "y--",
        label="normal height",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["critical height"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "r--",
        label="critical height",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "g",
        label="bathymetry",
    )
    if obs.shape[0] > 0:
        obs_bathymetry = bathymetry_interpolator(model.device, obs[:, :1])[0]
        ax.scatter(
            obs[:, 0].detach().clone().cpu().numpy(),
            obs[:, 1].detach().clone().cpu().numpy()
            + obs_bathymetry.flatten().detach().clone().cpu().numpy(),
            label="obs",
            color="#1f77b4",
        )

    ax.scatter(
        subdomains.detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].detach().clone().cpu().numpy()[indices],
        marker="|",
        c="k",
        s=100,
        label="subdomains",
    )
    ax.set_xlabel(r"$x \ [m]$")
    ax.set_ylabel(r"$y \ [m]$")
    ax.fill_between(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        0,
        color="green",
        alpha=0.3,
    )
    ax.fill_between(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        ref_solution["solution"].flatten().detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        color="blue",
        alpha=0.2,
    )
    ax.set_ylim(
        top=1.1 * max(ref_solution["bathymetry"] + ref_solution["solution"]).item(),
        bottom=0,
    )

    ax2 = ax.twinx()
    ax2.set_ylabel(r"$K_s \ [m^{1/3}/s]$", y=0.14)
    ax2.plot(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["parameter_function"].flatten().detach().clone().cpu().numpy(),
        "-.",
        label=r"$K_s^{true}(x)$",
        c="grey",
    )
    ax2.set_ylim(0, 300)
    ax2.set_yticks(np.arange(0, 100, 20))

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines + lines2,
        labels + labels2,
        loc="upper right",
        ncol=3,
        prop={"size": 7.5},
    )
    plt.show()


def display_results(model, col, ref_solution, obs, plot_col=False):
    subdomains, indices = _subdomain_indices(ref_solution)

    fig, ax = plt.subplots()
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["solution"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        color="#1f77b4",
        label="$h_{RK4}(x)$",
    )
    ax.plot(
        col.detach().clone().cpu().numpy(),
        model(normalize_input(col, col)).detach().clone().cpu().numpy()
        + ref_solution["bathymetry_col"].detach().clone().cpu().numpy(),
        "k--",
        label=r"$\tilde{h}(x)$",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "g",
        label="$b(x)$",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["normal height"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "y--",
        label="$h_n(x)$",
    )
    ax.plot(
        ref_solution["domain"].detach().clone().cpu().numpy(),
        ref_solution["critical height"].detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].detach().clone().cpu().numpy(),
        "r--",
        label="$h_c(x)$",
    )
    if obs.shape[0] > 0:
        obs_bathymetry = bathymetry_interpolator(model.device, obs[:, :1])[0]
        ax.scatter(
            obs[:, 0].detach().clone().cpu().numpy(),
            obs[:, 1].detach().clone().cpu().numpy()
            + obs_bathymetry.flatten().detach().clone().cpu().numpy(),
            label="obs",
            color="#1f77b4",
        )
    if plot_col:
        ax.scatter(
            col.detach().clone().cpu().numpy(),
            bathymetry_interpolator(model.device, col)[0]
            .detach()
            .clone()
            .cpu()
            .numpy(),
            label="col",
            color="black",
            s=10,
        )
    ax.scatter(
        subdomains.detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].detach().clone().cpu().numpy()[indices],
        marker="|",
        c="k",
        s=100,
        label="sub.",
    )
    ax.set_xlabel(r"$x \ [m]$")
    ax.set_ylabel(r"$y \ [m]$")
    ax.set_title("Calibrated model")
    ax.fill_between(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        0,
        color="green",
        alpha=0.3,
    )
    ax.fill_between(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        ref_solution["solution"].flatten().detach().clone().cpu().numpy()
        + ref_solution["bathymetry"].flatten().detach().clone().cpu().numpy(),
        color="blue",
        alpha=0.2,
    )
    ax.set_ylim(
        top=1.1 * max(ref_solution["bathymetry"] + ref_solution["solution"]).item(),
        bottom=0,
    )

    ax2 = ax.twinx()
    ax2.set_ylabel(r"$K_s \ [m^{1/3}/s]$", y=0.14)
    ax2.plot(
        ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
        ref_solution["parameter_function"].flatten().detach().clone().cpu().numpy(),
        "-.",
        label=r"$K_s^{true}(x)$",
        c="grey",
    )
    if hasattr(model, "parameter_values"):
        ax2.plot(
            ref_solution["domain"].flatten().detach().clone().cpu().numpy(),
            Ks_function(ref_solution["domain"], model.parameter_values(), col)
            .flatten()
            .detach()
            .clone()
            .cpu()
            .numpy(),
            "-.",
            label=r"$K_s(x)$",
            c="k",
        )
    ax2.set_ylim(0, 300)
    ax2.set_yticks(np.arange(0, 100, 20))

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines + lines2,
        labels + labels2,
        loc="upper right",
        ncol=2,
        prop={"size": 7.5},
    )
    plt.show()


def display_training(model, col, ref_solution):
    h_true = ref_solution["solution"].flatten().detach().clone().cpu().numpy()
    h_est = model(normalize_input(col, col)).detach().clone().cpu().numpy().flatten()
    rmse = np.linalg.norm(h_true - h_est, ord=2) / np.linalg.norm(h_true, ord=2)

    print("#" * 50)
    print(f"Final variable RMSE : {rmse:.2e}")
    print("#" * 50)
    return rmse
