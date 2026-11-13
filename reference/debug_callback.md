# Call an R function for debugging, also inside compiled code

Port of `jax.debug.callback`. `callback` is called with the arguments,
array leaves replaced by their values as R arrays, and its result is
ignored. Inside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) the call
happens when the compiled program runs; under
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) it is
called once per batch element; it is transparent to differentiation (it
sees primal values only). Called without `callback`, it returns a
function taking `(callback, ...)`, the two-call form of JAX.

## Usage

``` r
debug_callback(callback = NULL, ..., ordered = FALSE, partitioned = FALSE)
```

## Arguments

- callback:

  An R function.

- ...:

  Arguments passed to `callback` (pytrees; non-array leaves are passed
  as static values).

- ordered:

  Whether ordered debug callbacks keep their program order.

- partitioned:

  Ignored (single device).

## Value

`NULL`, invisibly (or a function in the two-call form).
