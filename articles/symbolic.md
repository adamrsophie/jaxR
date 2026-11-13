# Symbolic computation with symcompR

[symcompR](https://github.com/adamrsophie/symcompR) is a computer
algebra system for R. jaxR and symcompR are designed to be loaded
together: symcompR manipulates formulas exactly, jaxR evaluates and
differentiates them numerically on arrays. This vignette shows the three
points of contact. symcompR is an optional dependency; the code below
runs only when it is installed.

``` r

library(symcompR)
library(jaxR)
```

Both packages export `grad`, `hessian` and `jacobian`. Attach jaxR last,
as above: its versions handle functions themselves and pass symbolic
expressions on to symcompR, so one name serves both. (With the other
order, write
[`jaxR::grad`](https://adamrsophie.github.io/jaxR/reference/grad.md) for
functions.)

## From a formula to a compiled function

`jax_lambdify(args, expr)` converts a symbolic expression into a jaxR
function. The result is an ordinary jaxR function: it broadcasts over
arrays and can be passed to `jit`, `grad` and `vmap`.

``` r

x <- Symbol("x")
y <- Symbol("y")
expr <- sin(x) * y^2 + exp(x / y)
f <- jax_lambdify(list(x, y), expr)

f(as_jax(c(0.5, 1, 2)), as_jax(c(1, 2, 3)))
#> Array f32[3]
#> [1]  2.128147  5.014605 10.131411
jit(f)(as_jax(0.5), as_jax(1))
#> Array f32[] weak_type
#> [1] 2.128147
```

Elementary functions, special functions (`gamma`, `erf`, `digamma`,
`zeta`, Bessel functions, incomplete gamma functions), `Piecewise`,
relations, `Min`/`Max` and exact numbers are translated.

``` r

pw <- jax_lambdify(x, Piecewise(list(x^2, x < 1), list(2 * x, TRUE)))
pw(as_jax(c(0.5, 1, 2)))
#> Array f32[3]
#> [1] 0.25 2.00 4.00
```

## Two routes to a derivative

A symbolic derivative and an automatic derivative of the same formula
agree:

``` r

dfdx_symbolic <- jax_lambdify(list(x, y), doit(Derivative(expr, x)))
dfdx_auto <- grad(f)                      # with respect to the first argument

dfdx_symbolic(as_jax(0.5), as_jax(2))
#> Array f32[] weak_type
#> [1] 4.152343
dfdx_auto(as_jax(0.5), as_jax(2))
#> Array f32[] weak_type
#> [1] 4.152343
```

The symbolic route gives a formula that you can read and simplify. The
automatic route works for programs that have no closed form (loops,
solvers, random numbers) and scales to millions of parameters.

## One `grad` for both

jaxR’s `grad`, `hessian` and `jacobian` dispatch on the first argument:
a function is transformed, a symbolic expression is forwarded to
symcompR.

``` r

grad(x^2 * y, list(x, y))                 # symbolic gradient
#> Matrix(2, 1)
#> [2*x*y]
#> [x^2  ]
hessian(x^2 * y, list(x, y))              # symbolic Hessian matrix
#> Matrix(2, 2)
#> [2*y, 2*x]
#> [2*x, 0  ]
grad(function(v) v[0]^2 * v[1])(as_jax(c(3, 2)))    # numerical gradient
#> Array f32[2]
#> [1] 12  9
```

## Exact constants in array code

Symbolic numbers (integers, rationals, constants such as `pi`) combine
with arrays as weakly typed scalars. Expressions with free symbols are
rejected with a clear error.

``` r

as_jax(c(1, 2)) * Rational(1, 2)
#> Array f32[2]
#> [1] 0.5 1.0
```

## A worked example: Newton’s method from a formula

``` r

h <- x^3 - 2 * x - 5
hf <- jax_lambdify(x, h)
dhf <- jax_lambdify(x, doit(Derivative(h, x)))
newton <- jit(function(x0) {
  lax_fori_loop(0L, 20L, function(i, z) z - hf(z) / dhf(z), x0)
})
root <- newton(as_jax(2))
root
#> Array f32[] weak_type
#> [1] 2.094551
hf(root)
#> Array f32[] weak_type
#> [1] -1.907349e-06
```
