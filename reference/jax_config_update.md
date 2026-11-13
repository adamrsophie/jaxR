# Update a configuration option

Port of `jax.config.update`. The most important option is
`jax_enable_x64` (default `FALSE`, as in JAX: arrays default to 32-bit
types). The environment variable `JAX_ENABLE_X64` sets its initial
value.

## Usage

``` r
jax_config_update(name, value)
```

## Arguments

- name:

  Option name.

- value:

  New value.

## Value

The previous value, invisibly.
