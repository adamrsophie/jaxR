# lax special functions

Ports of the special functions of `jax.lax`: the regularized incomplete
beta integral (`lax_betainc`), the regularized incomplete gamma
functions and their derivatives (`lax_igamma`, `lax_igammac`,
`lax_igamma_grad_a`, `lax_random_gamma_grad`), the polygamma and Hurwitz
zeta functions and the exponentially scaled modified Bessel functions of
order 0 and 1. Arguments must be floating point arrays with equal dtypes
(R scalars adopt the dtype of the other arguments).

## Usage

``` r
lax_betainc(a, b, x)

lax_polygamma(m, x)

lax_igamma(a, x)

lax_igammac(a, x)

lax_igamma_grad_a(a, x)

lax_random_gamma_grad(a, x)

lax_zeta(x, q)

lax_bessel_i0e(x)

lax_bessel_i1e(x)
```

## Arguments

- a, b, x, m, q:

  Arrays.

## Value

An array.
