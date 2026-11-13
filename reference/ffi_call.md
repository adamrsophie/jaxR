# Call a foreign function

Port of `jax.ffi.ffi_call`. Returns a function that calls the XLA FFI
target `target_name` (registered with
[`ffi_register_target()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)).
Its unnamed arguments are the input arrays and its named arguments are
the attributes of the handler. It works eagerly and under
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) (see
`vmap_method`); it cannot be differentiated (use
[`custom_jvp()`](https://adamrsophie.github.io/jaxR/reference/custom_jvp.md)
or
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md)).

## Usage

``` r
ffi_call(
  target_name,
  result_shape_dtypes,
  has_side_effect = FALSE,
  vmap_method = NULL,
  input_layouts = NULL,
  output_layouts = NULL,
  input_output_aliases = NULL,
  custom_call_api_version = 4L,
  legacy_backend_config = NULL
)
```

## Arguments

- target_name:

  The name of the registered target.

- result_shape_dtypes:

  A
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  (or array) describing the output, or a list of them for several
  outputs (the function then returns a list).

- has_side_effect:

  Whether the call has side effects: it then runs even when its outputs
  are unused.

- vmap_method:

  How the call behaves under
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md), as
  for
  [`pure_callback()`](https://adamrsophie.github.io/jaxR/reference/pure_callback.md):
  `"sequential"`, `"sequential_unrolled"`, `"expand_dims"`,
  `"broadcast_all"`, `"legacy_vectorized"` or `NULL` (vmap is then an
  error).

- input_layouts, output_layouts:

  `NULL` (row major), or a list with one entry per input (output):
  `NULL` or a major to minor order of the axes (0-based, so `c(1, 0)` is
  a column major matrix).

- input_output_aliases:

  `NULL`, or a named integer vector or list mapping input indices
  (names) to output indices, 0-based: the output reuses the buffer of
  the input.

- custom_call_api_version:

  The custom call API of the target; 4 (the default) is the typed FFI.

- legacy_backend_config:

  For `custom_call_api_version < 4`: the opaque backend configuration
  string.

## Value

A function of the inputs and attributes.

## Details

Attributes convert as in JAX: an R integer scalar is a 64-bit integer (a
Python `int`), a double scalar a 64-bit float, a logical scalar a
boolean, a string a string, a jax scalar an attribute of its dtype, a
jax array or an R vector of length other than one a dense array (of
int32 for R integers and float64 for R doubles), a named list a
dictionary and an unnamed list an array of attributes.

## See also

[`ffi_register_target()`](https://adamrsophie.github.io/jaxR/reference/ffi_register_target.md)
