# Array API metadata

Ports of `jnp.__array_api_version__` and
`jnp.__array_namespace_info__()` (from
jax/\_src/numpy/array_api_metadata.py). The info object is a list of
functions: [`capabilities()`](https://rdrr.io/r/base/capabilities.html),
[`default_device()`](https://adamrsophie.github.io/jaxR/reference/default_device.md),
[`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md),
`default_dtypes(device)` and `dtypes(device, kind)`; `kind` is `NULL`
(all numeric types and bool), a kind name as in
[`jnp_isdtype()`](https://adamrsophie.github.io/jaxR/reference/jnp_dtype_functions.md),
or a vector of kind names.

## Usage

``` r
jnp_array_api_version

jnp_array_namespace_info()
```

## Value

`jnp_array_api_version` is a string; `jnp_array_namespace_info()`
returns the info list.
