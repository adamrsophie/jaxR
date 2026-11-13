# Mark values as used

Port of `jax.lax.dce_sink`: marks the computation of `val` as used so
that dead code elimination keeps it. jaxR programs have no side effects
that dead code elimination could drop, so this only checks that `val` is
a pytree of arrays.

## Usage

``` r
lax_dce_sink(val, prevent_mlir_dce = FALSE)
```

## Arguments

- val:

  A pytree of arrays.

- prevent_mlir_dce:

  Accepted for compatibility.

## Value

`NULL`, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_dce_sink(c(1, 2))
} # }
```
