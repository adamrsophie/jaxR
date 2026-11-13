# Hoist the closed-over values of a function

Port of `jax.closure_convert`: traces `fun` at the types of the example
arguments and hoists the closed-over values that may carry derivatives
(traced values) out of it. Returns `list(converted, consts)`;
`converted(..., <consts>)` takes the original arguments followed by the
hoisted values. Useful for writing
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)
rules of functions that close over traced values.

## Usage

``` r
closure_convert(fun, ...)
```

## Arguments

- fun:

  A function of pytrees of arrays.

- ...:

  Example arguments (only shapes and dtypes are used).

## Value

`list(converted, consts)`.
