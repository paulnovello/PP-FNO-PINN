import matplotlib.pyplot as plt
import numpy as np
import torch

from Backwater_model import Ks_function, bathymetry_interpolator
from normalization import normalize_input


def _to_numpy(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.numpy()


def _subdomain_indices(ref_solution):
    n_parameters = int(ref_solution["parameter"].numel())
    subdomains = torch.linspace(
        float(torch.min(ref_solution["domain"]).item()),
        float(torch.max(ref_solution["domain"]).item()),
        n_parameters,
    ).view(-1, 1)
    indices = (
        (subdomains / ref_solution["dx"])
        .clamp(max=ref_solution["domain"].shape[0] - 1)
        .int()
        .numpy()
    )
    return subdomains, indices


def display_data(col, ref_solution, obs, show=True):
    del col
    subdomains, indices = _subdomain_indices(ref_solution)

    fig, ax = plt.subplots()

    ax.set_title("Reference solution and data")
    domain = _to_numpy(ref_solution["domain"])
    surface = _to_numpy(ref_solution["solution"] + ref_solution["bathymetry"])
    critical_surface = _to_numpy(
        ref_solution["critical height"] + ref_solution["bathymetry"]
    )
    bathymetry = _to_numpy(ref_solution["bathymetry"])

    ax.plot(
        domain,
        surface,
        color="#1f77b4",
        label="RK4 solution",
    )
    ax.plot(
        domain,
        critical_surface,
        "r--",
        label="critical height",
    )
    ax.plot(
        domain,
        bathymetry,
        "g",
        label="bathymetry",
    )
    if obs.shape[0] > 0:
        obs_bathymetry = bathymetry_interpolator(obs[:, :1])[0]
        ax.scatter(
            _to_numpy(obs[:, 0]),
            _to_numpy(obs[:, 1]) + _to_numpy(obs_bathymetry.flatten()),
            label="obs",
            color="#1f77b4",
        )

    ax.scatter(
        _to_numpy(subdomains),
        bathymetry[indices],
        marker="|",
        c="k",
        s=100,
        label="subdomains",
    )
    ax.set_xlabel(r"$x \ [m]$")
    ax.set_ylabel(r"$y \ [m]$")
    ax.fill_between(
        domain.flatten(),
        bathymetry.flatten(),
        0,
        color="green",
        alpha=0.3,
    )
    ax.fill_between(
        domain.flatten(),
        bathymetry.flatten(),
        surface.flatten(),
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
        domain.flatten(),
        _to_numpy(ref_solution["parameter_function"]).flatten(),
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
    if show:
        plt.show()
        return None
    return fig


def display_results(model, col, ref_solution, obs, plot_col=False, show=True):
    subdomains, indices = _subdomain_indices(ref_solution)
    domain = _to_numpy(ref_solution["domain"])
    bathymetry = _to_numpy(ref_solution["bathymetry"])
    surface = _to_numpy(ref_solution["solution"] + ref_solution["bathymetry"])
    critical_surface = _to_numpy(
        ref_solution["critical height"] + ref_solution["bathymetry"]
    )
    with torch.no_grad():
        predicted_height = model(normalize_input(col, col))
    predicted_surface = _to_numpy(predicted_height + ref_solution["bathymetry_col"])

    fig, ax = plt.subplots()
    ax.plot(
        domain,
        surface,
        color="#1f77b4",
        label="$h_{RK4}(x)$",
    )
    ax.plot(
        _to_numpy(col),
        predicted_surface,
        "k--",
        label=r"$\tilde{h}(x)$",
    )
    ax.plot(
        domain,
        bathymetry,
        "g",
        label="$b(x)$",
    )
    ax.plot(
        domain,
        critical_surface,
        "r--",
        label="$h_c(x)$",
    )
    if obs.shape[0] > 0:
        obs_bathymetry = bathymetry_interpolator(obs[:, :1])[0]
        ax.scatter(
            _to_numpy(obs[:, 0]),
            _to_numpy(obs[:, 1]) + _to_numpy(obs_bathymetry.flatten()),
            label="obs",
            color="#1f77b4",
        )
    if plot_col:
        ax.scatter(
            _to_numpy(col),
            _to_numpy(bathymetry_interpolator(col)[0]),
            label="col",
            color="black",
            s=10,
        )
    ax.scatter(
        _to_numpy(subdomains),
        bathymetry[indices],
        marker="|",
        c="k",
        s=100,
        label="sub.",
    )
    ax.set_xlabel(r"$x \ [m]$")
    ax.set_ylabel(r"$y \ [m]$")
    ax.set_title("Calibrated model")
    ax.fill_between(
        domain.flatten(),
        bathymetry.flatten(),
        0,
        color="green",
        alpha=0.3,
    )
    ax.fill_between(
        domain.flatten(),
        bathymetry.flatten(),
        surface.flatten(),
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
        domain.flatten(),
        _to_numpy(ref_solution["parameter_function"]).flatten(),
        "-.",
        label=r"$K_s^{true}(x)$",
        c="grey",
    )
    if hasattr(model, "parameter_values"):
        with torch.no_grad():
            ks_values = Ks_function(ref_solution["domain"], model.parameter_values(), col)
        ax2.plot(
            domain.flatten(),
            _to_numpy(ks_values).flatten(),
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
    if show:
        plt.show()
        return None
    return fig


def display_training(model, col, ref_solution):
    h_true = _to_numpy(ref_solution["solution"]).flatten()
    with torch.no_grad():
        h_est = _to_numpy(model(normalize_input(col, col))).flatten()
    rmse = np.linalg.norm(h_true - h_est, ord=2) / np.linalg.norm(h_true, ord=2)

    print("#" * 50)
    print(f"Final variable RMSE : {rmse:.2e}")
    print("#" * 50)
    return rmse
