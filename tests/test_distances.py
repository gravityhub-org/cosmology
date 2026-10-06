"""Checks against Adachi & Kasai (2012) exact limits and numerical integrals."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from cosmology.distances import (
    C_KM_S,
    angular_diameter_distance,
    dyer_roeder_angular_diameter_distance,
    dyer_roeder_luminosity_distance,
    hubble_parameter,
    luminosity_distance,
    rho_crit,
)


def _num_dL(z, omega_m, H0=70.0):
    def integrand(a):
        return 1.0 / np.sqrt(omega_m * a + (1.0 - omega_m) * a**4)

    I, _ = quad(integrand, 1.0 / (1.0 + z), 1.0, epsabs=1e-10)
    return (C_KM_S / H0) * (1.0 + z) * I


def test_omega_m_1_exact():
    z = 1.5
    H0 = 70.0
    expected = (2.0 * C_KM_S / H0) * (1.0 + z) * (1.0 - 1.0 / np.sqrt(1.0 + z))
    got = float(luminosity_distance(jnp.asarray(z), omega_m=1.0, H0=H0))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_reciprocity():
    z = jnp.asarray(0.8)
    dL = luminosity_distance(z)
    dA = angular_diameter_distance(jnp.asarray(0.0), z)
    np.testing.assert_allclose(float(dL), float((1.0 + z) ** 2 * dA), rtol=1e-12)


def test_empty_beam_reciprocity():
    z = jnp.asarray(0.8)
    DL = dyer_roeder_luminosity_distance(z)
    DA = dyer_roeder_angular_diameter_distance(jnp.asarray(0.0), z)
    np.testing.assert_allclose(float(DL), float((1.0 + z) ** 2 * DA), rtol=1e-12)


@pytest.mark.parametrize("z", [0.03, 0.1, 1.0, 10.0])
def test_filled_beam_vs_integral_omega_m_0p3(z):
    approx = float(luminosity_distance(jnp.asarray(z), omega_m=0.3))
    numeric = _num_dL(z, 0.3)
    # Paper Table 1: at Ωm=0.3 the relative error is ≲ 0.26% for these z.
    np.testing.assert_allclose(approx, numeric, rtol=4e-3)


def test_hubble_today():
    np.testing.assert_allclose(float(hubble_parameter(jnp.asarray(0.0))), 70.0)


def test_rho_crit_positive():
    assert float(rho_crit(jnp.asarray(0.5))) > float(rho_crit(jnp.asarray(0.0)))
