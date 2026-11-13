# Call an impure R function from compiled code

Port of `jax.experimental.io_callback`. Like
[`pure_callback()`](https://adamrsophie.github.io/jaxR/reference/pure_callback.md),
but the callback is treated as having side effects: it is not removed
when its results are unused, and it cannot be differentiated. Under
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) it is
called once per batch element (an `ordered` callback cannot be vmapped).
`ordered = TRUE` orders the ordered callbacks of one compiled region
with a token.

## Usage

``` r
experimental_io_callback(
  callback,
  result_shape_dtypes,
  ...,
  sharding = NULL,
  ordered = FALSE
)

io_callback(
  callback,
  result_shape_dtypes,
  ...,
  sharding = NULL,
  ordered = FALSE
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

- ordered:

  Whether calls must keep their program order.

## Value

A pytree of arrays with the structure of `result_shape_dtypes`.
