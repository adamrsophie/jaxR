# Gamma family random values

Ports of `jax.random.gamma`, `loggamma`, `beta`, `dirichlet`,
`chisquare`, `f` and `t`. The exact samplers are rejection samplers
(Marsaglia and Tsang); their gradients with respect to the shape
parameters use implicit reparameterization through
[`lax_random_gamma_grad()`](https://adamrsophie.github.io/jaxR/reference/lax_special.md).
`random_random_gamma_p` is the underlying primitive.

## Usage

``` r
random_gamma(
  key,
  a,
  shape = NULL,
  dtype = NULL,
  method = "exact",
  out_sharding = NULL
)

random_loggamma(
  key,
  a,
  shape = NULL,
  dtype = NULL,
  method = "exact",
  out_sharding = NULL
)

random_random_gamma_p

random_beta(
  key,
  a,
  b,
  shape = NULL,
  dtype = NULL,
  method = "exact",
  out_sharding = NULL
)

random_dirichlet(key, alpha, shape = NULL, dtype = NULL, out_sharding = NULL)

random_chisquare(
  key,
  df,
  shape = NULL,
  dtype = NULL,
  method = "exact",
  out_sharding = NULL
)

random_f(key, dfnum, dfden, shape = NULL, dtype = NULL, out_sharding = NULL)

random_t(key, df, shape = NULL, dtype = NULL, out_sharding = NULL)
```

## Arguments

- key:

  A PRNG key from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- a, b:

  Shape parameters (`a` for gamma, `a` and `b` for beta).

- shape:

  Integer vector; `NULL` broadcasts the parameters.

- dtype:

  A float dtype.

- method:

  `"exact"` (rejection sampler) or `"approximate"` (loop free, slightly
  biased).

- out_sharding:

  Ignored (one device).

- alpha:

  Dirichlet concentration of shape `(..., n)`.

- df:

  Degrees of freedom.

- dfnum, dfden:

  Numerator and denominator degrees of freedom.

## Value

An array.
