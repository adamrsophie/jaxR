# Check derivatives against finite differences

Ports of `jax.test_util.check_grads`, `check_jvp` and `check_vjp`.
Derivatives are compared with a central finite difference in one random
direction, so the cost does not grow with the number of inputs or
outputs. `test_util_check_grads()` checks forward and reverse mode up to
`order`.

## Usage

``` r
test_util_check_grads(
  f,
  args,
  order,
  modes = c("fwd", "rev"),
  atol = NULL,
  rtol = NULL,
  eps = NULL
)

test_util_check_jvp(
  f,
  f_jvp,
  args,
  atol = NULL,
  rtol = NULL,
  eps = 1e-04,
  err_msg = ""
)

test_util_check_vjp(
  f,
  f_vjp,
  args,
  atol = NULL,
  rtol = NULL,
  eps = 1e-04,
  err_msg = ""
)
```

## Arguments

- f:

  Function to check at `do.call(f, args)`.

- args:

  List of argument values.

- order:

  Derivatives up to this order are checked.

- modes:

  Modes to check: any of `"fwd"`, `"rev"` and `"lin"`.

- atol, rtol:

  Absolute and relative tolerance.

- eps:

  Step of the finite difference (default `1e-4`).

- f_jvp:

  Function of `(primals, tangents)` computing the JVP of `f`, typically
  `function(p, t) jvp(f, p, t)`.

- err_msg:

  Text added to the error message.

- f_vjp:

  Function of the arguments computing the VJP of `f`, typically
  `function(...) vjp(f, ...)`.

## Value

`TRUE`, invisibly; an error of class `jax_assertion_error` when a check
fails.

## Details

Tolerances default to JAX's per dtype gradient tolerances (`2e-3` for
`float32`, `1e-5` for `float64`); as in JAX they are multiplied by the
number of elements compared. Pass one number, or numbers named by dtype
to override single dtypes. The random direction is drawn with a fixed
seed and the caller's random number stream is left untouched.

## Examples

``` r
if (FALSE) { # \dontrun{
test_util_check_grads(function(x) sin(x) * x, list(as_jax(c(1, 2, 3))), order = 2)
} # }
```
