# Rematerialization decisions of checkpoint policies

Ports of `jax.ad_checkpoint.Saveable`, `Recompute` and `Offloadable`,
the values a policy of
[`checkpoint()`](https://adamrsophie.github.io/jaxR/reference/checkpoint.md)
may return besides `TRUE` and `FALSE`: save the value, recompute it, or
offload it from memory space `src` to `dst` (jaxR has one memory space,
so offloading saves). `Saveable` and `Offloadable()` count as `TRUE` and
`Recompute` as `FALSE` (they are classed logical values).

## Usage

``` r
Saveable

Recompute

Offloadable(src, dst)
```

## Format

`Saveable` and `Recompute` are classed logical scalars.

## Arguments

- src, dst:

  Memory kinds (for example `"device"` and `"pinned_host"`).

## Examples

``` r
policy <- function(prim, params) if (prim == "sin") Saveable else Recompute
f <- checkpoint(function(x) sin(sin(x)), policy = policy)
```
