# Markers for the `with_refs` method of VJP functions

Ports of `jax.ad.GradValue`, `jax.ad.DontWant`, `jax.ad.GradRef` and
`jax.ad.DidntWant`. The arguments of `vjp_fun$with_refs(...)` (see
[`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md)) have the
structure of the primal arguments; each leaf is a
[`new_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md) (the
gradient is added to it in place), `ad_GradValue()` (the gradient is
returned, the default) or `ad_DontWant()` (the gradient is not
computed). The VJP function then returns `ad_GradRef()` for the leaves
accumulated into a ref and `ad_DidntWant()` for the leaves not wanted.

## Usage

``` r
ad_GradValue()

ad_DontWant()

ad_GradRef()

ad_DidntWant()
```

## Value

A marker object.

## Examples

``` r
if (FALSE) { # \dontrun{
r <- vjp(function(x, y) x * y, 3, 4)
ref <- new_ref(as_jax(0, "float32"))
r[[2]]$with_refs(ad_DontWant(), ref)(1)
ref[]  # 3
} # }
```
