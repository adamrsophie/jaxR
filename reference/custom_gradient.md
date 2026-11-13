# Define a function with a custom gradient

Port of `jax.custom_gradient`: `fun` returns `list(value, vjp)` where
`vjp(g)` maps the output cotangent to the input cotangents (a list with
one entry per argument, or the cotangent itself when `fun` has a single
argument). Built on
[`custom_vjp()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp.md);
the values `vjp` closes over are saved as residuals.

## Usage

``` r
custom_gradient(fun, with_logs = FALSE, with_accums = NULL)
```

## Arguments

- fun:

  A function returning `list(value, vjp)`.

- with_logs:

  Whether `vjp` also returns logs.

- with_accums:

  `NULL`, or the accumulator kinds `vjp` receives.

## Value

A function returning `value`, differentiated with `vjp`.

## Details

With `with_logs = TRUE`, `vjp(g)` returns `list(in_cts, logs)`, where
`logs` is a named list of values to log out of the backward pass or
`NULL` (see
[`custom_vjp_defvjp_with_logs()`](https://adamrsophie.github.io/jaxR/reference/custom_vjp_defvjp_with_logs.md)).
With `with_accums`, `vjp(g, ...)` takes one gradient accumulator per
argument (with the structure of the arguments) instead of returning
cotangents, and returns `NULL` or a named list of logs. `with_accums`
declares the kind of each accumulator: a tree prefix of the list of
arguments with leaves
[ad_ValAccum](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md)
or
[ad_RefAccum](https://adamrsophie.github.io/jaxR/reference/ad_ValAccum.md).
Arguments that are not differentiated (including integer ones) get a
NullAccum whatever is declared.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- custom_gradient(function(x) list(x^2, function(g) g * x))
grad(f)(3)  # 3, not 6
f <- custom_gradient(function(x) list(x^2, function(g) list(2 * g * x, list(ct = g))),
                     with_logs = TRUE)
vjp(f, 3)[[2]]$with_logs(1)
} # }
```
