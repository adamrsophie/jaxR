# Name a value for checkpoint policies (forward pass)

Port of `jax.ad_checkpoint.checkpoint_name_fwd`; without the
experimental remat3 mode of JAX it is
[`checkpoint_name()`](https://adamrsophie.github.io/jaxR/reference/checkpoint_name.md).

## Usage

``` r
checkpoint_name_fwd(x, name)
```

## Arguments

- x:

  A pytree of arrays.

- name:

  Name used by
  [checkpoint_policies](https://adamrsophie.github.io/jaxR/reference/checkpoint_policies.md).

## Value

`x`.

## Examples

``` r
if (FALSE) { # \dontrun{
checkpoint_name_fwd(c(1, 2), "a")
} # }
```
