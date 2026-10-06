r"""Flat LCDM distances: Adachi & Kasai (2012), arXiv:1111.6396.

Filled-beam d_L, d_A(z1, z2) and empty-beam (Dyer-Roeder) D_L, D_A(z1, z2)
use the Pade fits in that paper. H(z) and rho_c(z) are exact.
"""

import jax.numpy as jnp

C_KM_S = 299792.458
MPC_M = 3.0856775814913673e22
G_SI = 6.67430e-11
DEFAULT_H0 = 70.0
DEFAULT_OMEGA_M = 0.3


def _x(z, omega_m):
    """Eq. (3): x(z, Omega_m) = (1 - Omega_m) / Omega_m / (1 + z)^3."""
    return (1.0 - omega_m) / omega_m / (1.0 + z) ** 3


def _phi(x):
    """Eq. (2): filled-beam rational function Phi(x)."""
    return (1.0 + 1.320 * x + 0.4415 * x**2 + 0.02656 * x**3) / (
        1.0 + 1.392 * x + 0.5121 * x**2 + 0.03944 * x**3
    )


def _psi(x):
    """Eq. (19): empty-beam rational function Psi(x)."""
    return (1.0 + 1.256 * x + 0.3804 * x**2 + 0.0164 * x**3) / (
        1.0 + 1.483 * x + 0.6072 * x**2 + 0.0587 * x**3
    )


def luminosity_distance(z, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0):
    """Filled-beam luminosity distance d_L(z) in Mpc. Eq. (1)."""
    return (
        (2.0 * C_KM_S / H0)
        * (1.0 + z)
        / jnp.sqrt(omega_m)
        * (_phi(_x(0.0, omega_m)) - _phi(_x(z, omega_m)) / jnp.sqrt(1.0 + z))
    )


def angular_diameter_distance(z1, z2, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0):
    """Filled-beam angular diameter distance d_A(z1, z2) in Mpc. Eq. (22)."""
    return (
        (2.0 * C_KM_S / H0)
        / (jnp.sqrt(omega_m) * (1.0 + z2))
        * (
            _phi(_x(z1, omega_m)) / jnp.sqrt(1.0 + z1)
            - _phi(_x(z2, omega_m)) / jnp.sqrt(1.0 + z2)
        )
    )


def dyer_roeder_luminosity_distance(z, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0):
    """Empty-beam Dyer-Roeder luminosity distance D_L(z) in Mpc. Eq. (18)."""
    return (
        (2.0 / 5.0)
        * (C_KM_S / H0)
        * (1.0 + z) ** 2
        / jnp.sqrt(omega_m)
        * (
            _psi(_x(0.0, omega_m))
            - _psi(_x(z, omega_m)) / (1.0 + z) ** 2.5
        )
    )


def dyer_roeder_angular_diameter_distance(
    z1, z2, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0
):
    """Empty-beam Dyer-Roeder D_A(z1, z2) in Mpc. Eq. (24)."""
    return (
        (2.0 / 5.0)
        * (C_KM_S / H0)
        * (1.0 + z1)
        / jnp.sqrt(omega_m)
        * (
            _psi(_x(z1, omega_m)) / (1.0 + z1) ** 2.5
            - _psi(_x(z2, omega_m)) / (1.0 + z2) ** 2.5
        )
    )


def hubble_parameter(z, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0):
    """H(z) in km/s/Mpc for flat LCDM (exact; paper appendix B eq. 31)."""
    return H0 * jnp.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def rho_crit(z, omega_m=DEFAULT_OMEGA_M, H0=DEFAULT_H0):
    """Critical density rho_c(z) = 3 H(z)^2 / (8 pi G) in kg/m^3."""
    H_si = hubble_parameter(z, omega_m=omega_m, H0=H0) * 1000.0 / MPC_M
    return 3.0 * H_si**2 / (8.0 * jnp.pi * G_SI)
