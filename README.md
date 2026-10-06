# cosmology

JAX cosmological distance measures for a spatially flat dust + cosmological-constant universe, following Adachi & Kasai ([arXiv:1111.6396](https://arxiv.org/abs/1111.6396)).

Filled-beam and empty-beam (Dyer–Roeder) luminosity and angular-diameter distances use their rational fits (no integrals). Comoving distances and volumes are algebraic in those distances; \(H(z)\) and \(\rho_{\mathrm{c}}(z)\) use the closed-form Hubble function.

## Install

```bash
uv add "cosmology @ git+https://github.com/gravityhub-org/cosmology.git"
```

## Usage

```python
import jax.numpy as jnp
from cosmology import luminosity_distance, angular_diameter_distance, comoving_distance

z = jnp.asarray(0.5)
print(luminosity_distance(z))          # Mpc; H0=70, Om0=0.3
print(angular_diameter_distance(z))    # Mpc
print(comoving_distance(z))            # Mpc
```

Default cosmology: \(H_0=70\,\mathrm{km\,s^{-1}\,Mpc^{-1}}\), \(\Omega_{\mathrm{m}}=0.3\), \(\Omega_\Lambda=1-\Omega_{\mathrm{m}}\).

## Tests

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
