# Call a pure R function from compiled code

Port of `jax.pure_callback`. `callback` is called with R arrays (the
host values of the array arguments, as
[`as.array()`](https://rdrr.io/r/base/array.html) returns them) and must
return R data or jax arrays matching `result_shape_dtypes`. It works
under [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) and the
other transformations: inside compiled code the call goes through an XLA
FFI handler that runs the function on the R main thread. The callback is
assumed pure: it may be called several times, or not at all when its
result is unused.

## Usage

``` r
pure_callback(
  callback,
  result_shape_dtypes,
  ...,
  sharding = NULL,
  vmap_method = NULL
)
```

## Arguments

- callback:

  An R function.

- result_shape_dtypes:

  A pytree of
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  values (or arrays) describing the results.

- ...:

  Arguments (pytrees of arrays) passed to `callback`; named arguments
  are passed by name.

- sharding:

  Ignored (single device).

- vmap_method:

  One of `"sequential"`, `"sequential_unrolled"`, `"expand_dims"`,
  `"broadcast_all"`, `"legacy_vectorized"` or `NULL`.

## Value

A pytree of arrays with the structure of `result_shape_dtypes`.

## Details

Under [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md)
the behavior follows `vmap_method`: `"sequential"` (and
`"sequential_unrolled"`) loops over the batch with
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md);
`"expand_dims"` adds a leading axis of size one to unbatched arguments;
`"broadcast_all"` broadcasts them to the batch size;
`"legacy_vectorized"` passes batched arguments unchanged. With
`vmap_method = NULL`, vmap raises an error, as in JAX. Differentiation
raises an error unless the callback is wrapped in
[`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
or
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md).

## Examples

``` r
if (FALSE) { # \dontrun{
f <- function(x) pure_callback(function(v) besselJ(v, 0), ShapeDtypeStruct(dim(x), "float32"), x)
jit(f)(as_jax(c(0.5, 1, 2)))
} # }
```
