# Geo encoders

Geophysical inputs usually arrive as longitude/latitude in degrees, while downstream neural-field and GP features typically want periodic encodings or unit-sphere coordinates. The geo encoders in `pyrox_nn` make those preprocessing steps first-class and composable.

The canonical spherical-harmonic pipeline is:

```python
import equinox as eqx

from pyrox_nn import (
    Cartesian3DEncoder,
    Deg2Rad,
    SphericalHarmonicEncoder,
)

encoder = eqx.nn.Sequential(
    [
        Deg2Rad(),
        Cartesian3DEncoder(input_unit="radians"),
        SphericalHarmonicEncoder(l_max=8, input_mode="cartesian"),
    ]
)
features = encoder(lonlat_deg)  # (N, 81)
```

`Cartesian3DEncoder` uses the same axis convention expected by `pyrox_gp.SphericalHarmonicInducingFeatures`, so the NN and GP spherical paths line up. For temporal complements, see `fourier_features` and `seasonal_features`.

## Stateful encoder layers

::: pyrox_nn.Deg2Rad

::: pyrox_nn.LonLatScale

::: pyrox_nn.Cartesian3DEncoder

::: pyrox_nn.CyclicEncoder

::: pyrox_nn.SphericalHarmonicEncoder

## Slepian encoders

Region-localized spherical encoders built on the Slepian concentration
problem. The deterministic `SlepianEncoder` and
`HybridSphericalSlepianEncoder` are re-exported from `geonnax`;
`BayesianSlepianEncoder` adds NumPyro sites over the cap radius and
centre.

::: pyrox_nn.SlepianEncoder

::: pyrox_nn.HybridSphericalSlepianEncoder

::: pyrox_nn.BayesianSlepianEncoder

## Pure-JAX helper functions

::: pyrox_nn.deg2rad

::: pyrox_nn.lonlat_scale

::: pyrox_nn.lonlat_to_cartesian3d

::: pyrox_nn.cyclic_encode

::: pyrox_nn.spherical_harmonic_encode
