# Serialize a compiled function

Ports of `jax.experimental.serialize_executable.serialize` and
`deserialize_and_load`. `experimental_serialize_executable_serialize()`
takes the result of `jit(f)$lower(...)$compile()` and returns
`list(payload, in_tree, out_tree)`: `payload` is a raw vector with the
executable and its constants, which can be saved and loaded again by a
later R session on the same platform with the same XLA runtime.
`deserialize_and_load()` rebuilds the compiled function from the three
parts.

## Usage

``` r
experimental_serialize_executable_serialize(compiled)

experimental_serialize_executable_deserialize_and_load(
  serialized,
  in_tree,
  out_tree
)
```

## Arguments

- compiled:

  A compiled function from `$compile()`.

- serialized:

  The payload.

- in_tree, out_tree:

  The trees returned with the payload.

## Value

`serialize`: `list(payload, in_tree, out_tree)`. `deserialize_and_load`:
a compiled function.

## Details

As with JAX, the payload is trusted: loading it runs native code and R's
[`unserialize()`](https://rdrr.io/r/base/serialize.html). Only load
payloads you made yourself.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) x * 2)
compiled <- f$lower(as_jax(c(1, 2)))$compile()
s <- experimental_serialize_executable_serialize(compiled)
g <- experimental_serialize_executable_deserialize_and_load(s[[1]], s[[2]], s[[3]])
g(as_jax(c(3, 4)))
} # }
```
