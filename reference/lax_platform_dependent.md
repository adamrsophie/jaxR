# Platform dependent code

Port of `jax.lax.platform_dependent`. jaxR runs on the CPU, so the `cpu`
branch (or `default` when there is none) is applied to the arguments.

## Usage

``` r
lax_platform_dependent(..., default = NULL)
```

## Arguments

- ...:

  Arguments passed to the selected branch (unnamed) and the branches,
  named by platform (`cpu`, `cuda`, `rocm`, `tpu`, ...).

- default:

  Optional branch for platforms without their own branch.

## Value

The result of the selected branch.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_platform_dependent(2, cpu = function(x) x + 1, default = function(x) x - 1)
} # }
```
