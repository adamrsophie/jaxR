# Continuous random values

Ports of `jax.random.cauchy`, `exponential`, `gumbel`, `laplace`,
`logistic`, `pareto`, `rayleigh`, `wald`, `maxwell`,
`double_sided_maxwell`, `weibull_min`, `triangular`,
`generalized_normal`, `ball` and `orthogonal`.

## Usage

``` r
random_cauchy(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_exponential(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_gumbel(
  key,
  shape = integer(0),
  dtype = NULL,
  mode = NULL,
  out_sharding = NULL
)

random_laplace(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_logistic(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_pareto(key, b, shape = NULL, dtype = NULL, out_sharding = NULL)

random_rayleigh(key, scale, shape = NULL, dtype = NULL, out_sharding = NULL)

random_wald(key, mean, shape = NULL, dtype = NULL, out_sharding = NULL)

random_maxwell(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_double_sided_maxwell(key, loc, scale, shape = integer(0), dtype = NULL)

random_weibull_min(key, scale, concentration, shape = integer(0), dtype = NULL)

random_triangular(
  key,
  left,
  mode,
  right,
  shape = NULL,
  dtype = NULL,
  out_sharding = NULL
)

random_generalized_normal(
  key,
  p,
  shape = integer(0),
  dtype = NULL,
  out_sharding = NULL
)

random_ball(
  key,
  d,
  p = 2,
  shape = integer(0),
  dtype = NULL,
  out_sharding = NULL
)

random_orthogonal(
  key,
  n,
  shape = integer(0),
  dtype = NULL,
  m = NULL,
  out_sharding = NULL
)
```

## Arguments

- key:

  A PRNG key from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- shape:

  Integer vector; `NULL` (where allowed) broadcasts the parameters.

- dtype:

  A float dtype.

- out_sharding:

  Ignored (one device).

- mode:

  For `random_gumbel()`, the bits used: `"low"` (default), `"high"` or
  `"highest"`; for `random_triangular()`, the peak of the distribution.

- b:

  Pareto shape parameter.

- scale:

  Scale parameter.

- mean:

  Wald mean.

- loc:

  Location parameter.

- concentration:

  Weibull concentration.

- left, right:

  Triangular limits.

- p:

  Shape parameter (`generalized_normal`), or the `p` of the Lp ball.

- d:

  Dimension of the ball.

- n, m:

  Rows and columns of the orthogonal matrix (`m` defaults to `n`).

## Value

An array.
