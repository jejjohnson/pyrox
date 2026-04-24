# Geo encoders

Geophysical inputs usually arrive as longitude/latitude in degrees, while downstream neural-field and GP features typically want periodic encodings or unit-sphere coordinates. The geo encoders in `pyrox.nn` make those preprocessing steps first-class and composable.

The canonical spherical-harmonic pipeline is:

```python
import equinox as eqx

from pyrox.nn import (
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

`Cartesian3DEncoder` uses the same axis convention expected by `pyrox.gp.SphericalHarmonicInducingFeatures`, so the NN and GP spherical paths line up. For temporal complements, see `fourier_features` and `seasonal_features`.

## Stateful encoder layers

::: pyrox.nn.Deg2Rad

::: pyrox.nn.LonLatScale

::: pyrox.nn.Cartesian3DEncoder

::: pyrox.nn.CyclicEncoder

::: pyrox.nn.SphericalHarmonicEncoder

## Pure-JAX helper functions

::: pyrox.nn.deg2rad

::: pyrox.nn.lonlat_scale

::: pyrox.nn.lonlat_to_cartesian3d

::: pyrox.nn.cyclic_encode

::: pyrox.nn.spherical_harmonic_encode
