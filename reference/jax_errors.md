# Error classes

Port of `jax.errors`. Every error raised by jaxR is an R condition of
class `jax_error`; the more specific classes below correspond to the
exception classes of JAX and can be used in
[`tryCatch()`](https://rdrr.io/r/base/conditions.html) handlers. The
exported `errors_*` values are the class names as strings, so that
`inherits(e, errors_TracerBoolConversionError)` reads like the Python
`isinstance` test.

## Usage

``` r
errors_JAXTypeError

errors_JAXIndexError

errors_ConcretizationTypeError

errors_NonConcreteBooleanIndexError

errors_TracerArrayConversionError

errors_TracerBoolConversionError

errors_TracerIntegerConversionError

errors_UnexpectedTracerError

errors_KeyReuseError

errors_JaxRuntimeError
```

## Format

Character strings.

## Details

|  |  |  |
|----|----|----|
| JAX exception | Condition class | Raised when |
| `JAXTypeError` | `jax_type_error` | an argument has the wrong type or dtype |
| `JAXIndexError` | `jax_index_error` | an index is invalid |
| `ConcretizationTypeError` | `jax_concretization_error` | a traced value is used where a concrete value is needed |
| `TracerArrayConversionError` | `jax_tracer_array_conversion_error` | [`as.array()`](https://rdrr.io/r/base/array.html), [`as.vector()`](https://rdrr.io/r/base/vector.html) or [`as.numeric()`](https://rdrr.io/r/base/numeric.html) is called on a traced value |
| `TracerBoolConversionError` | `jax_tracer_bool_conversion_error` | [`as.logical()`](https://rdrr.io/r/base/logical.html) is called on a traced value (for example in `if`) |
| `TracerIntegerConversionError` | `jax_tracer_integer_conversion_error` | [`as.integer()`](https://rdrr.io/r/base/integer.html) is called on a traced value |
| `NonConcreteBooleanIndexError` | `jax_non_concrete_boolean_index_error` | a traced logical array is used as an index |
| `UnexpectedTracerError` | `jax_unexpected_tracer_error` | a traced value leaked out of its transformation |
| `KeyReuseError` | `jax_key_reuse_error` | a random key is used twice while key reuse checking is on |
| `JaxRuntimeError` | `jax_runtime_error` | the XLA runtime reports an error |

The tracer conversion classes also inherit from
`jax_concretization_error`, and all of the classes that derive from
`JAXTypeError` in JAX inherit from `jax_type_error`. Other Python
exception types map to `jax_value_error` (`ValueError`),
`jax_not_implemented_error` (`NotImplementedError`), `jax_key_error`,
`jax_attribute_error`, `jax_assertion_error`, `jax_overflow_error` and
`jax_floating_point_error`.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) if (as.logical(x > 0)) x else -x)
tryCatch(f(1), jax_tracer_bool_conversion_error = function(e) "use lax_cond() instead")
} # }
```
