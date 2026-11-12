# Port of jax.errors: the condition classes signaled by jaxR.

#' Error classes
#'
#' Port of `jax.errors`. Every error raised by jaxR is an R condition of class
#' `jax_error`; the more specific classes below correspond to the exception classes of
#' JAX and can be used in [tryCatch()] handlers. The exported `errors_*` values are the
#' class names as strings, so that `inherits(e, errors_TracerBoolConversionError)` reads
#' like the Python `isinstance` test.
#'
#' | JAX exception | Condition class | Raised when |
#' |---|---|---|
#' | `JAXTypeError` | `jax_type_error` | an argument has the wrong type or dtype |
#' | `JAXIndexError` | `jax_index_error` | an index is invalid |
#' | `ConcretizationTypeError` | `jax_concretization_error` | a traced value is used where a concrete value is needed |
#' | `TracerArrayConversionError` | `jax_tracer_array_conversion_error` | `as.array()`, `as.vector()` or `as.numeric()` is called on a traced value |
#' | `TracerBoolConversionError` | `jax_tracer_bool_conversion_error` | `as.logical()` is called on a traced value (for example in `if`) |
#' | `TracerIntegerConversionError` | `jax_tracer_integer_conversion_error` | `as.integer()` is called on a traced value |
#' | `NonConcreteBooleanIndexError` | `jax_non_concrete_boolean_index_error` | a traced logical array is used as an index |
#' | `UnexpectedTracerError` | `jax_unexpected_tracer_error` | a traced value leaked out of its transformation |
#' | `KeyReuseError` | `jax_key_reuse_error` | a random key is used twice while key reuse checking is on |
#' | `JaxRuntimeError` | `jax_runtime_error` | the XLA runtime reports an error |
#'
#' The tracer conversion classes also inherit from `jax_concretization_error`, and all
#' of the classes that derive from `JAXTypeError` in JAX inherit from `jax_type_error`.
#' Other Python exception types map to `jax_value_error` (`ValueError`),
#' `jax_not_implemented_error` (`NotImplementedError`), `jax_key_error`,
#' `jax_attribute_error`, `jax_assertion_error`, `jax_overflow_error` and
#' `jax_floating_point_error`.
#' @format Character strings.
#' @examples
#' \dontrun{
#' f <- jit(function(x) if (as.logical(x > 0)) x else -x)
#' tryCatch(f(1), jax_tracer_bool_conversion_error = function(e) "use lax_cond() instead")
#' }
#' @name jax_errors
NULL

#' @rdname jax_errors
#' @export
errors_JAXTypeError <- "jax_type_error"
#' @rdname jax_errors
#' @export
errors_JAXIndexError <- "jax_index_error"
#' @rdname jax_errors
#' @export
errors_ConcretizationTypeError <- "jax_concretization_error"
#' @rdname jax_errors
#' @export
errors_NonConcreteBooleanIndexError <- "jax_non_concrete_boolean_index_error"
#' @rdname jax_errors
#' @export
errors_TracerArrayConversionError <- "jax_tracer_array_conversion_error"
#' @rdname jax_errors
#' @export
errors_TracerBoolConversionError <- "jax_tracer_bool_conversion_error"
#' @rdname jax_errors
#' @export
errors_TracerIntegerConversionError <- "jax_tracer_integer_conversion_error"
#' @rdname jax_errors
#' @export
errors_UnexpectedTracerError <- "jax_unexpected_tracer_error"
#' @rdname jax_errors
#' @export
errors_KeyReuseError <- "jax_key_reuse_error"
#' @rdname jax_errors
#' @export
errors_JaxRuntimeError <- "jax_runtime_error"
