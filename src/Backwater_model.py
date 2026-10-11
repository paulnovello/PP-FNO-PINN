"""1D backwater physical model.

Defines the physical constants (gravity ``g``, flow rate ``q``, downstream
boundary condition ``h_BC``, critical height ``hc``), the bathymetry loaded
from ``bathy.npy``, the parametrization of the Strickler coefficient
``K_s(x)``, and an RK4 solver that produces the reference solution.
"""

from pathlib import Path

import numpy as np
import torch


g = 9.81
q = 1
h_BC = 1
regime = "subcritical"
hc = (q**2 / g) ** (1 / 3)

BASE_DIR = Path(__file__).resolve().parent

bathy = torch.tensor(np.load(BASE_DIR / "bathy.npy")) / 5
bathy_prime = bathy[1:] - bathy[:-1]
bathy_prime = torch.hstack((bathy_prime, bathy_prime[-1]))
bathy_x = torch.arange(bathy.shape[0])


def domain_bounds(col):
    """Return the bounds of a 1D grid.

    Args:
        col: tensor of coordinates, any shape.

    Returns:
        Tuple ``(x_min, x_max)`` of Python floats.
    """
    return float(torch.min(col).item()), float(torch.max(col).item())


def numpy_interpolator(x, domain, y):
    """Linearly interpolate ``y(domain)`` at the points ``x`` with ``np.interp``.

    The result is detached from the autograd graph.

    Args:
        x: query coordinates, any shape (flattened).
        domain: increasing coordinates where ``y`` is known.
        y: values at ``domain``.

    Returns:
        Tensor of shape ``[len(x), 1]``.
    """
    interpolated = np.interp(
        x.detach().clone().cpu().numpy().flatten(),
        domain.detach().clone().cpu().numpy().flatten(),
        y.detach().clone().cpu().numpy().flatten(),
    )
    return torch.from_numpy(interpolated).view(-1, 1)


def bathymetry_interpolator(x):
    """Evaluate the bathymetry and its slope at ``x``.

    Args:
        x: coordinates in meters, any shape (flattened).

    Returns:
        Tuple ``(b, b_prime)`` of tensors of shape ``[N, 1]``: the bed
        elevation ``b(x)`` and its slope ``b'(x)`` (precomputed by finite
        differences on the 1 m data grid). Neither carries gradients.
    """
    b = numpy_interpolator(x, bathy_x, bathy)
    b_prime = numpy_interpolator(x, bathy_x, bathy_prime)
    return b, b_prime


def Ks_function(x, k, col):
    """Build the spatially varying Strickler coefficient ``K_s(x)``.

    The domain spanned by ``col`` is split into ``len(k) - 1`` equal
    subdomains, and ``K_s`` is piecewise linear between the values ``k``
    placed at the subdomain boundaries. A single value gives a constant
    ``K_s``. The output is differentiable with respect to ``k``.

    Args:
        x: coordinates where ``K_s`` is evaluated, shape ``[N, 1]``.
        k: tensor of ``p`` control values of ``K_s``.
        col: reference grid that defines the domain bounds.

    Returns:
        Tensor of shape ``[N, 1]``.
    """
    k = k.view(-1)
    if k.numel() == 1:
        return k.view(1, 1).expand(x.shape[0], 1)

    x_min, x_max = domain_bounds(col)
    subdomains = torch.linspace(x_min, x_max, k.numel())
    subdomain_sizes = subdomains[1:] - subdomains[:-1]
    flat_x = x.view(-1)

    indices = (torch.bucketize(flat_x, subdomains) - 1).clamp(
        min=0,
        max=k.numel() - 2,
    )
    alpha = (flat_x - subdomains[indices]) / subdomain_sizes[indices]
    values = k[indices + 1] * alpha + k[indices] * (1 - alpha)
    return values.view(-1, 1)


def compute_ref_solution(col, k, dx):
    """Compute the reference water height by integrating Eq. (1) with RK4.

    Integration starts at the downstream boundary (``h = h_BC``) and
    marches upstream with step ``dx``. Only the subcritical regime is
    supported.

    Args:
        col: collocation grid; only its bounds are used.
        k: control values of ``K_s`` (see ``Ks_function``).
        dx: integration step in meters.

    Returns:
        Dict with keys ``"solution"`` (h on ``"domain"``), ``"domain"``,
        ``"dx"``, ``"critical height"``, ``"normal height"``,
        ``"bathymetry"``, ``"bathymetry_col"`` (bathymetry on ``col``),
        ``"parameter"`` (``k``), ``"parameter_function"`` (``K_s`` on
        ``"domain"``) and ``"regime"``.

    Raises:
        Warning: if the solution drops below the critical height.
    """
    if regime != "subcritical":
        raise NotImplementedError("Only the subcritical regime is supported.")

    x_min, x_max = domain_bounds(col)
    domain = torch.linspace(
        x_min,
        x_max,
        int((x_max - x_min) / dx),
    ).view(-1, 1)

    bathymetry = bathymetry_interpolator(domain)[0]
    slope = -bathymetry_interpolator(domain)[1]
    k = k.float()

    def backwater_model(x, h, parameter):
        """Right-hand side of Eq. (1), signed for upstream integration."""
        froude = q / (g * h**3) ** (1 / 2)
        return -(
            numpy_interpolator(x, domain, slope)
            - (q / parameter) ** 2 / h ** (10 / 3)
        ) / (1 - froude**2)

    def rk4_integrator(parameter_values):
        """March RK4 upstream; return ``(h, normal_height)`` as column arrays."""
        index = domain.shape[0] - 1
        list_h = [h_BC]
        list_hn = [
            (
                (
                    q**2
                    / (
                        numpy_interpolator(domain[index], domain, slope)
                        * Ks_function(domain[index], parameter_values, col) ** 2
                    )
                )
                ** (3 / 10)
            ).item()
        ]

        while index > 0:
            k1 = backwater_model(
                domain[index],
                list_h[-1],
                Ks_function(domain[index], parameter_values, col),
            )
            k2 = backwater_model(
                domain[index] - dx / 2,
                list_h[-1] - dx / 2 * k1,
                Ks_function(domain[index], parameter_values, col),
            )
            k3 = backwater_model(
                domain[index] - dx / 2,
                list_h[-1] - dx / 2 * k2,
                Ks_function(domain[index], parameter_values, col),
            )
            k4 = backwater_model(
                domain[index] - dx,
                list_h[-1] - dx * k3,
                Ks_function(domain[index], parameter_values, col),
            )

            list_h.append((list_h[-1] + dx / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).item())

            local_slope = numpy_interpolator(domain[index], domain, slope)
            if local_slope < 0:
                list_hn.append(np.nan)
            else:
                list_hn.append(
                    (
                        (
                            q**2
                            / (
                                local_slope
                                * Ks_function(domain[index], parameter_values, col) ** 2
                            )
                        )
                        ** (3 / 10)
                    ).item()
                )

            if list_h[-1] < hc:
                raise Warning("You reached supercritical regime !")

            index -= 1

        return np.flip(np.array(list_h, dtype=np.float32)).reshape(-1, 1), np.flip(
            np.array(list_hn, dtype=np.float32)
        ).reshape(-1, 1)

    solution, normal_height = rk4_integrator(k)

    h = torch.tensor(solution.copy())
    h_n = torch.tensor(normal_height.copy())
    h_c = hc * torch.ones(domain.shape[0], 1)

    return {
        "solution": h,
        "dx": dx,
        "critical height": h_c,
        "normal height": h_n,
        "bathymetry": bathymetry,
        "bathymetry_col": bathymetry_interpolator(col)[0],
        "domain": domain,
        "parameter": k,
        "parameter_function": Ks_function(domain, k, col),
        "regime": regime,
    }
