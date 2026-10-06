# cosmology

JAX cosmological distance measures for a spatially flat dust + cosmological-constant universe, following Adachi & Kasai, *PASJ* **64**, 7 (2012) ([arXiv:1111.6396](https://arxiv.org/abs/1111.6396)).

Filled-beam and empty-beam (Dyer–Roeder) luminosity and angular-diameter distances use their rational Padé fits. The Hubble parameter and critical density are the exact flat-ΛCDM expressions.

## Install

```bash
uv add "cosmology @ git+https://github.com/gravityhub-org/cosmology.git"
```

## Usage

```python
import jax.numpy as jnp
from cosmology import (
    luminosity_distance,
    angular_diameter_distance,
    hubble_parameter,
    rho_crit,
)

z = jnp.asarray(0.5)
print(luminosity_distance(z))                    # Mpc; H0=70, Omega_m=0.3
print(angular_diameter_distance(0.0, z))         # Mpc, observer → source
print(hubble_parameter(z))                       # km/s/Mpc
print(rho_crit(z))                               # kg/m^3
```

Distances are in Mpc for `H0` in km/s/Mpc. Default cosmology is \(H_0=70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}\), \(\Omega_{\mathrm{m}}=0.3\), \(\Omega_\Lambda=1-\Omega_{\mathrm{m}}\).

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
