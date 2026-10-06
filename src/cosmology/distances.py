r"""Flat LCDM distances in JAX: Adachi & Kasai (2012), arXiv:1111.6396.

Filled-beam and empty-beam (Dyer-Roeder) distances are the paper's rational
fits — no numerical integrals. Other measures are algebraic in those
distances and in the closed-form Hubble function E(z).
"""

import jax.numpy as jnp

C_KM_S = 299792.458
MPC_M = 3.0856775814913673e22
G_SI = 6.67430e-11
DEFAULT_H0 = 70.0
DEFAULT_OM0 = 0.3


def _x(z, Om0):
    """Eq. (3): x(z, Om0) = ((1 - Om0) / Om0) / (1 + z)^3."""
    return (1.0 - Om0) / Om0 / (1.0 + z) ** 3


def _phi(x):
    """Eq. (2): filled-beam Phi(x)."""
    return (1.0 + 1.320 * x + 0.4415 * x**2 + 0.02656 * x**3) / (
        1.0 + 1.392 * x + 0.5121 * x**2 + 0.03944 * x**3
    )


def _psi(x):
    """Eq. (19): empty-beam Psi(x)."""
    return (1.0 + 1.256 * x + 0.3804 * x**2 + 0.0164 * x**3) / (
        1.0 + 1.483 * x + 0.6072 * x**2 + 0.0587 * x**3
    )


def _E(z, Om0):
    """Dimensionless Hubble function, flat LCDM (appendix B eq. 31)."""
    return jnp.sqrt(Om0 * (1.0 + z) ** 3 + (1.0 - Om0))


def hubble_parameter(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """H(z) in km/s/Mpc."""
    return H0 * _E(z, Om0)


def luminosity_distance(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Filled-beam luminosity distance d_L(z) in Mpc. Eq. (1)."""
    z = jnp.asarray(z)
    return (
        (2.0 * C_KM_S / H0)
        * (1.0 + z)
        / jnp.sqrt(Om0)
        * (_phi(_x(0.0, Om0)) - _phi(_x(z, Om0)) / jnp.sqrt(1.0 + z))
    )


def angular_diameter_distance_z1z2(z1, z2, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Filled-beam d_A(z1, z2) in Mpc. Eq. (22)."""
    z1 = jnp.asarray(z1)
    z2 = jnp.asarray(z2)
    return (
        (2.0 * C_KM_S / H0)
        / (jnp.sqrt(Om0) * (1.0 + z2))
        * (
            _phi(_x(z1, Om0)) / jnp.sqrt(1.0 + z1)
            - _phi(_x(z2, Om0)) / jnp.sqrt(1.0 + z2)
        )
    )


def angular_diameter_distance(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Filled-beam d_A(0, z) in Mpc."""
    return angular_diameter_distance_z1z2(0.0, z, H0=H0, Om0=Om0)


def comoving_transverse_distance(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Transverse comoving distance D_M = (1 + z) d_A in Mpc (flat)."""
    return (1.0 + z) * angular_diameter_distance(z, H0=H0, Om0=Om0)


def comoving_distance(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Line-of-sight comoving distance D_C in Mpc (equals D_M when flat)."""
    return comoving_transverse_distance(z, H0=H0, Om0=Om0)


def differential_comoving_volume(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """dV_c / dz / dOmega in Mpc^3 / sr."""
    z = jnp.asarray(z)
    DH = C_KM_S / H0
    dm = comoving_transverse_distance(z, H0=H0, Om0=Om0)
    return DH * dm**2 / _E(z, Om0)


def comoving_volume(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Comoving volume out to z in Mpc^3 (full sky, flat)."""
    dm = comoving_transverse_distance(z, H0=H0, Om0=Om0)
    return (4.0 * jnp.pi / 3.0) * dm**3


def dyer_roeder_luminosity_distance(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Empty-beam Dyer-Roeder D_L(z) in Mpc. Eq. (18)."""
    z = jnp.asarray(z)
    return (
        (2.0 / 5.0)
        * (C_KM_S / H0)
        * (1.0 + z) ** 2
        / jnp.sqrt(Om0)
        * (_psi(_x(0.0, Om0)) - _psi(_x(z, Om0)) / (1.0 + z) ** 2.5)
    )


def dyer_roeder_angular_diameter_distance(
    z1, z2, H0=DEFAULT_H0, Om0=DEFAULT_OM0
):
    """Empty-beam Dyer-Roeder D_A(z1, z2) in Mpc. Eq. (24)."""
    z1 = jnp.asarray(z1)
    z2 = jnp.asarray(z2)
    return (
        (2.0 / 5.0)
        * (C_KM_S / H0)
        * (1.0 + z1)
        / jnp.sqrt(Om0)
        * (
            _psi(_x(z1, Om0)) / (1.0 + z1) ** 2.5
            - _psi(_x(z2, Om0)) / (1.0 + z2) ** 2.5
        )
    )


def rho_crit(z, H0=DEFAULT_H0, Om0=DEFAULT_OM0):
    """Critical density rho_c(z) = 3 H(z)^2 / (8 pi G) in kg/m^3."""
    H_si = hubble_parameter(z, H0=H0, Om0=Om0) * 1000.0 / MPC_M
    return 3.0 * H_si**2 / (8.0 * jnp.pi * G_SI)
