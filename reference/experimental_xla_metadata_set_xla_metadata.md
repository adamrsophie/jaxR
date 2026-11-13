# Attach XLA metadata to operations

Ports of `jax.experimental.xla_metadata`. XLA metadata are frontend
attributes (string key value pairs) on the lowered operations, which XLA
reads as hints, for example `FUSE_LIMIT` (a fusion boundary) or
`MUST_FUSE` (fuse a group).

## Usage

``` r
experimental_xla_metadata_set_xla_metadata(x = NULL, ..., code)

experimental_xla_metadata_fuse_limit(code)

experimental_xla_metadata_must_fuse_call(identifier)

experimental_xla_metadata_xla_metadata_call(f = NULL, ...)

experimental_xla_metadata_xla_metadata_call2(
  f = NULL,
  metadata = NULL,
  ad_metadata = "same"
)
```

## Arguments

- x:

  A pytree of arrays to tag, or `NULL`.

- ...:

  Metadata as named arguments.

- code:

  An expression evaluated with the metadata in effect.

- identifier:

  The fusion group.

- f:

  A function, or `NULL` for a decorator.

- metadata:

  A named list of metadata.

- ad_metadata:

  `"same"` (the default), `"drop"`, or a named list: the metadata of the
  computations autodiff derives from `f` (see Details).

## Value

The tagged `x`, the value of `code`, or a function.

## Details

- `experimental_xla_metadata_set_xla_metadata(x, ...)` tags the
  operations that produce the arrays of the pytree `x` and returns `x`.

- `experimental_xla_metadata_set_xla_metadata(..., code = )` evaluates
  `code` with the metadata in effect: every operation staged meanwhile
  (under [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)
  and the other transformations that compile) is tagged. Nested uses
  merge, the inner value of a key winning; a value of `NULL` removes a
  key. Without `x` and `code` it returns a function that wraps a
  function to run with the metadata in effect (the decorator form of
  JAX).

- `experimental_xla_metadata_fuse_limit(code)` evaluates `code` with
  `FUSE_LIMIT = TRUE`.

- `experimental_xla_metadata_must_fuse_call(identifier)` returns a
  decorator that tags the operations of a function with
  `MUST_FUSE = identifier`.

- `experimental_xla_metadata_xla_metadata_call(f, ...)` and
  `experimental_xla_metadata_xla_metadata_call2(f, metadata, ad_metadata)`
  wrap `f` so that its operations carry the metadata.

Values may be strings, logicals or numbers; they are attached as strings
(logicals as `"true"` and `"false"`, doubles as Python writes floats).
Unlike JAX, which stages `xla_metadata_call` as a call operation tagged
as a whole, jaxR tags the operations of the function directly, which XLA
treats the same once it inlines the call; and the operations of the
backward pass of
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) are
tagged only when they are staged inside the context (`ad_metadata` other
than `"drop"` is not applied).

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) {
  y <- experimental_xla_metadata_set_xla_metadata(jnp_sin(x), my_key = "a")
  experimental_xla_metadata_fuse_limit(y * 2)
})
cat(jit_lower(f, jnp_ones(3))$as_text())
} # }
```
