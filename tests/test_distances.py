"""Adachi & Kasai (2012) vs Astropy LambdaCDM (Tcmb0=0) and JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.cosmology import LambdaCDM

from cosmology import (
    angular_diameter_distance,
    angular_diameter_distance_z1z2,
    comoving_distance,
    comoving_transverse_distance,
    comoving_volume,
    differential_comoving_volume,
    dyer_roeder_angular_diameter_distance,
    dyer_roeder_luminosity_distance,
    hubble_parameter,
    luminosity_distance,
    rho_crit,
)
from cosmology.distances import C_KM_S

H0 = 70.0
OM0 = 0.3
# Matter + Lambda only, matching Adachi's flat LCDM (no radiation).
ASTROPY = LambdaCDM(H0=H0, Om0=OM0, Ode0=1.0 - OM0, Tcmb0=0.0)
# Paper Table 1: Om0=0.3 relative error is 0.26% at z=0.03, smaller at higher z.
# Volumes scale as D_M^2 or D_M^3 so the same d_A error is amplified.
DISTANCE_RTOL = 3e-3
VOLUME_RTOL = 1e-2
Z = jnp.asarray([0.03, 0.1, 0.5, 1.0, 3.0, 10.0])


def test_omega_m_1_exact():
    z = jnp.asarray(1.5)
    expected = (2.0 * C_KM_S / H0) * (1.0 + z) * (1.0 - 1.0 / jnp.sqrt(1.0 + z))
    np.testing.assert_allclose(
        luminosity_distance(z, H0=H0, Om0=1.0), expected, rtol=1e-12
    )


def test_reciprocity():
    z = jnp.asarray(0.8)
    dL = luminosity_distance(z)
    dA = angular_diameter_distance(z)
    np.testing.assert_allclose(dL, (1.0 + z) ** 2 * dA, rtol=1e-12)


def test_empty_beam_reciprocity():
    z = jnp.asarray(0.8)
    DL = dyer_roeder_luminosity_distance(z)
    DA = dyer_roeder_angular_diameter_distance(0.0, z)
    np.testing.assert_allclose(DL, (1.0 + z) ** 2 * DA, rtol=1e-12)


def test_luminosity_distance_vs_astropy():
    got = np.asarray(luminosity_distance(Z, H0=H0, Om0=OM0))
    want = ASTROPY.luminosity_distance(np.asarray(Z)).to_value(u.Mpc)
    np.testing.assert_allclose(got, want, rtol=DISTANCE_RTOL)


def test_angular_diameter_distance_vs_astropy():
    got = np.asarray(angular_diameter_distance(Z, H0=H0, Om0=OM0))
    want = ASTROPY.angular_diameter_distance(np.asarray(Z)).to_value(u.Mpc)
    np.testing.assert_allclose(got, want, rtol=DISTANCE_RTOL)


def test_angular_diameter_distance_z1z2_vs_astropy():
    z1 = jnp.asarray(0.4)
    z2 = jnp.asarray(1.2)
    got = float(angular_diameter_distance_z1z2(z1, z2, H0=H0, Om0=OM0))
    want = ASTROPY.angular_diameter_distance(0.4, 1.2).to_value(u.Mpc)
    np.testing.assert_allclose(got, want, rtol=DISTANCE_RTOL)


def test_comoving_distance_vs_astropy():
    got = np.asarray(comoving_distance(Z, H0=H0, Om0=OM0))
    want = ASTROPY.comoving_distance(np.asarray(Z)).to_value(u.Mpc)
    np.testing.assert_allclose(got, want, rtol=DISTANCE_RTOL)


def test_comoving_transverse_distance_vs_astropy():
    got = np.asarray(comoving_transverse_distance(Z, H0=H0, Om0=OM0))
    want = ASTROPY.comoving_transverse_distance(np.asarray(Z)).to_value(u.Mpc)
    np.testing.assert_allclose(got, want, rtol=DISTANCE_RTOL)


def test_differential_comoving_volume_vs_astropy():
    got = np.asarray(differential_comoving_volume(Z, H0=H0, Om0=OM0))
    want = ASTROPY.differential_comoving_volume(np.asarray(Z)).to_value(
        u.Mpc**3 / u.sr
    )
    np.testing.assert_allclose(got, want, rtol=VOLUME_RTOL)


def test_comoving_volume_vs_astropy():
    got = np.asarray(comoving_volume(Z, H0=H0, Om0=OM0))
    want = ASTROPY.comoving_volume(np.asarray(Z)).to_value(u.Mpc**3)
    np.testing.assert_allclose(got, want, rtol=VOLUME_RTOL)


def test_hubble_parameter_vs_astropy():
    got = np.asarray(hubble_parameter(Z, H0=H0, Om0=OM0))
    want = ASTROPY.H(np.asarray(Z)).to_value(u.km / u.s / u.Mpc)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_rho_crit_vs_astropy():
    got = np.asarray(rho_crit(Z, H0=H0, Om0=OM0))
    want = ASTROPY.critical_density(np.asarray(Z)).to_value(u.kg / u.m**3)
    np.testing.assert_allclose(got, want, rtol=1e-4)


def test_jit_vmap_grad():
    z = jnp.asarray(0.7)
    zs = jnp.linspace(0.1, 2.0, 8)
    dL = jax.jit(luminosity_distance)
    np.testing.assert_allclose(dL(z), luminosity_distance(z), rtol=1e-12)
    np.testing.assert_allclose(
        jax.vmap(luminosity_distance)(zs), luminosity_distance(zs), rtol=1e-12
    )
    g = jax.grad(lambda zi: luminosity_distance(zi))(z)
    assert jnp.isfinite(g)
    assert g > 0
