# Gradient accumulators

The kinds of gradient accumulator a backward rule can receive (ports of
`ValAccum`, `RefAccum` and `NullAccum` of JAX's autodiff). A rule
defined with
[`custom_vjp_defvjp_with_accums()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)
or `custom_gradient(with_accums = )` gets one accumulator per argument
leaf and pushes cotangent contributions into it:

## Usage

``` r
ad_ValAccum

ad_RefAccum

ad_NullAccum
```

## Format

Marker objects.

## Details

- a `ValAccum` collects values: `acc$accum(x)` adds `x`;

- a `RefAccum` holds a ref, `acc$ref`, to add to in place
  (`acc$ref[i] <- acc$ref[i] + g`); `acc$accum(x)` adds to all of it;

- a `NullAccum` (for an argument that is not differentiated) ignores
  what is pushed.

`ad_ValAccum` and `ad_RefAccum` are the kinds that
`custom_gradient(with_accums = )` declares; `acc$aval` is the type of
the cotangent.
