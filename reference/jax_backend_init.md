# Initialize the XLA backend

Called automatically on first use.

## Usage

``` r
jax_backend_init(plugin = find_pjrt_plugin())
```

## Arguments

- plugin:

  Path to a PJRT plugin shared library.

## Value

Backend information (platform, number of devices, PJRT API version).
