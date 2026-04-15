---
status: draft
version: 0.1.0
---

# Primitives — Core Abstractions (`pyrox._core`)

## `PyroxModule` (`eqx.Module` subclass)

| Method / Attribute | Description |
|---|---|
| `pyrox_param(name, init_value, *, constraint, event_dim)` | Register/retrieve a deterministic learnable parameter via `numpyro.param`. Supports constraints (e.g., `positive`) with automatic bijective transforms. |
| `pyrox_sample(name, prior)` | Register/retrieve a random variable via `numpyro.sample`. Accepts `Distribution`, callable `(self) → Distribution`, or plain tensor (`numpyro.deterministic`). |
| `_get_context()` | Return (or lazily create) the per-instance `_Context` cache |
| `_pyro_fullname(name)` | Produce a fully-qualified site name scoped by the module hierarchy |
| `setup()` | Subclass hook for eager attribute registration |


## `PyroxParam` (`NamedTuple`)

| Field | Type | Description |
|---|---|---|
| `init_value` | `ndarray \| Callable \| None` | Initial value, lazy callable, or `None` (look up existing) |
| `constraint` | `Constraint` | Parameter domain constraint (default: `real`) |
| `event_dim` | `int \| None` | Number of rightmost event dimensions |

Can be used inline in `__call__` or as a class-variable default.


## `PyroxSample` (frozen `dataclass`)

| Field | Type | Description |
|---|---|---|
| `prior` | `Distribution \| Callable` | Independent prior, or callable `(self) → Distribution` for dependent priors |

Supports dependent priors: `lambda self_: dist.LogNormal(self_.loc, 0.1)`.


## `Parameterized` (extends `PyroxModule`)

| Method | Description |
|---|---|
| `register_param(name, init_value, constraint)` | Declare a constrained parameter with initial value |
| `set_prior(name, distribution)` | Attach a prior distribution to a registered parameter |
| `autoguide(name, guide_type)` | Set guide type for a parameter: `"delta"` (MAP), `"normal"` (mean-field), `"mvn"` (full covariance) |
| `set_mode(mode)` | Switch between `"model"` (sample from priors) and `"guide"` (sample from learned variational dists) |
| `get_param(name)` | Retrieve parameter value — dispatches based on current mode and whether a prior/guide is set |
| `load_pyro_samples()` | Coordinate with NumPyro trace during inference — triggers sample/param sites for all registered parameters |
| `_teardown()` | Clean up class-level registries to prevent memory leaks |

Per-instance registry stores: priors, guide types, guide parameters, mode flag. Keyed by `id(self)`.


## `_Context` (internal)

| Method | Description |
|---|---|
| `__enter__` / `__exit__` | Context manager with re-entrant nesting depth counter |
| `get(name)` | Return cached value or `None` |
| `set(name, value)` | Store value in cache (no-op when inactive) |

Cache is cleared when the outermost scope exits. Prevents duplicate site registration.


## Utilities

| Export | Type | Description |
|---|---|---|
| `pyrox_method` | decorator | Gives non-`__call__` methods their own context scope |
| `_biject_to(constraint)` | function | Returns bijective transform from unconstrained → constrained space |
| `_is_real_support(support)` | function | Check whether a constraint is effectively unconstrained |
