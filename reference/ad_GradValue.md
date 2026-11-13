# Markers for the `with_refs` method of VJP functions

Ports of `jax.ad.GradValue`, `jax.ad.DontWant`, `jax.ad.GradRef` and
`jax.ad.DidntWant`. The arguments of `vjp_fun$with_refs(...)` (see
[`vjp()`](https://adamrsophie.github.io/jaxR/reference/vjp.md)) have the
structure of the primal arguments; each leaf is a
[`new_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md) (the
gradient is added to it), `ad_GradValue()` (the gradient is returned,
the default) or `ad_DontWant()` (the gradient is not returned). Inside a
backward rule a leaf can also be one of the accumulators the rule
received (see
[ad_ValAccum](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)):
the gradient is pushed into it. The VJP function then returns
`ad_GradRef()` for the leaves accumulated into a ref and
`ad_DidntWant()` for the leaves not wanted.

## Usage

``` r
ad_GradValue()

ad_DontWant()

ad_GradRef()

ad_DidntWant()
```

## Value

A marker object.

## Details

Unlike JAX, jaxR computes every gradient and adds it into the ref after
the backward pass, and custom rules receive the accumulator kind they
declare whatever the caller asks for (see
[ad_ValAccum](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)).

## Examples

``` r
if (FALSE) { # \dontrun{
r <- vjp(function(x, y) x * y, 3, 4)
ref <- new_ref(as_jax(0, "float32"))
r[[2]]$with_refs(ad_DontWant(), ref)(1)
ref[]  # 3
} # }
```
