# The configuration object

Port of `jax.config`: `config$update(name, value)` sets an option (as
[`jax_config_update()`](https://adamrsophie.github.io/jaxR/reference/jax_config_update.md)),
`config$read(name)` reads one, `config$values` lists all of them, and
`config$<option>` (for example `config$jax_enable_x64`) reads a single
option.

## Usage

``` r
config
```

## Format

An object of class `jax_config_object`.
