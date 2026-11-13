# Runtime error checks that work under jit, vmap and control flow

Port of `jax.experimental.checkify`. A compiled function cannot raise an
R error that depends on the values it computes.
`experimental_checkify_checkify()` turns such checks into data: the
checked function returns an error value next to its result, and the
error is raised (or inspected) outside the compiled code.

## Usage

``` r
experimental_checkify_user_checks

experimental_checkify_nan_checks

experimental_checkify_index_checks

experimental_checkify_div_checks

experimental_checkify_float_checks

experimental_checkify_automatic_checks

experimental_checkify_all_checks

experimental_checkify_JaxRuntimeError

experimental_checkify_init_error

experimental_checkify_Error

experimental_checkify_checkify(f, errors = experimental_checkify_user_checks)

experimental_checkify_check(pred, msg, ..., debug = FALSE)

experimental_checkify_debug_check(pred, msg, ...)

experimental_checkify_check_error(error)
```

## Arguments

- f:

  Function to check.

- errors:

  Character vector of error categories, one of the sets above or several
  combined with [`c()`](https://rdrr.io/r/base/c.html).

- pred:

  Scalar boolean: the check fails if it is `FALSE`.

- msg:

  Message of the error, a format string.

- ...:

  Formatting arguments of `msg`: arrays or pytrees of arrays, unnamed
  for [`{}`](https://rdrr.io/r/base/Paren.html) and `{0}`, named for
  `{name}`.

- debug:

  If `TRUE` the check only runs under
  `experimental_checkify_checkify()`.

- error:

  An error value returned by a checked function.

## Value

`experimental_checkify_checkify()`: a function returning
`list(error, value)`. The check functions return `NULL` invisibly.

## Details

- `experimental_checkify_check(pred, msg, ...)` adds a check to a
  function: an error with the message `msg` if `pred` is `FALSE`. Called
  outside any transformation it raises at once. Inside
  [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
  [`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
  and the other staging functions it must be functionalized by
  `experimental_checkify_checkify()` first.

- `experimental_checkify_debug_check()` is a check that only runs under
  `experimental_checkify_checkify()` and is dropped otherwise.

- `experimental_checkify_checkify(f, errors)` returns a function with
  the arguments of `f` that returns `list(error, value)`. `errors`
  selects the checks: the user checks (the default) and the automatic
  checks listed below. Sets are combined with
  [`c()`](https://rdrr.io/r/base/c.html).

- `experimental_checkify_check_error(error)` raises the error of an
  error value. It is itself a check, so it can be functionalized again
  by an outer `experimental_checkify_checkify()`.

The error value (`Error` in JAX) is a pytree, so it passes through
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) and
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md). Its
methods are `error$get()` (the message of the first error, or `NULL`),
`error$get_exception()` and `error$throw()` (raise the error, like
`experimental_checkify_check_error()`). After
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) the
message lists the mapped indices that failed.
`experimental_checkify_init_error` is the empty error value.

The sets of checks:

|  |  |
|----|----|
| set | checks |
| `experimental_checkify_user_checks` | calls of `experimental_checkify_check()` |
| `experimental_checkify_nan_checks` | a floating point operation produced a NaN |
| `experimental_checkify_div_checks` | a division by zero |
| `experimental_checkify_index_checks` | an index out of bounds |
| `experimental_checkify_float_checks` | NaN and division checks |
| `experimental_checkify_automatic_checks` | float and index checks |
| `experimental_checkify_all_checks` | automatic and user checks |

Raised errors have the class `experimental_checkify_JaxRuntimeError`
(`"jax_checkify_runtime_error"`, a `jax_value_error`). Error values have
the class `experimental_checkify_Error` (`"jax_checkify_error"`).

With
`jax_config_update("jax_experimental_unsafe_xla_runtime_errors", TRUE)`
a check that was not functionalized is compiled to a host callback that
raises the error while the compiled code runs.

Messages are format strings in the style of Python:
[`{}`](https://rdrr.io/r/base/Paren.html) and `{0}` take the unnamed
arguments in `...`, `{name}` a named one, and a printf like
specification may follow a colon (`{x:.3f}`). The values may be traced
arrays; they are formatted by R when the message is built.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- function(x) {
  experimental_checkify_check(x > 0, "{x} needs to be positive!", x = x)
  1 / x
}
checked <- jit(experimental_checkify_checkify(f))
res <- checked(as_jax(-3))
res[[1]]$get()

g <- function(x) x + sin(x)
res <- experimental_checkify_checkify(g, errors = experimental_checkify_float_checks)(Inf)
res[[1]]$get()
} # }
```
