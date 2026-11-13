# Taylor mode higher order automatic differentiation

Port of `jax.experimental.jet.jet`. Where
[`jvp()`](https://adamrsophie.github.io/jaxR/reference/jvp.md) pushes
one tangent through a function, `experimental_jet_jet()` pushes a
truncated Taylor polynomial: given the value `h0` of an inner function
and its derivatives `h1, ..., hK` along a direction, it returns the
value of `fun(h)` and its derivatives of order 1 to K, without nesting
first order differentiation K times.

## Usage

``` r
experimental_jet_jet(fun, primals, series, factorial_scaled = TRUE)
```

## Arguments

- fun:

  Function of one or more arrays, returning an array or a pytree of
  arrays.

- primals:

  List of the arguments of `fun`: the point of expansion.

- series:

  List with one entry per argument; each entry is the list of the higher
  order terms of that argument. All entries have the same length, the
  order of the expansion.

- factorial_scaled:

  If `TRUE` (the default) the terms of the input and of the output are
  derivatives. If `FALSE` they are Taylor coefficients, the derivatives
  divided by the factorial of their order.

## Value

`list(primals_out, series_out)`: the value of `fun` and, with the same
structure, the list of the higher order terms of each output.

## Details

Rules exist for the primitives that JAX has rules for (arithmetic, the
elementary functions, linear structural operations, reductions,
`dot_general`, convolutions, gathers and scatter-add). As in JAX,
control flow primitives have no rule, and custom derivative rules are
ignored: the function itself is differentiated.

A rule for another primitive is a function
`function(primals_in, series_in, params)` assigned to the `jet` field of
the primitive object (the counterpart of `jet.jet_rules[prim] = rule`);
it returns `list(primal_out, series_out)`.

## Examples

``` r
if (FALSE) { # \dontrun{
# f(h(z)) = sin(z^3) at z = 0.5: h and its first two derivatives go in.
h <- list(0.5^3, 3 * 0.5^2, 6 * 0.5)
out <- experimental_jet_jet(jnp_sin, list(h[[1]]), list(h[2:3]))
out[[1]]        # sin(h0)
out[[2]][[1]]   # cos(h0) * h1
out[[2]][[2]]   # -sin(h0) * h1^2 + cos(h0) * h2
} # }
```
