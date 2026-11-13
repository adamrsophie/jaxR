# Normal random values

Ports of `jax.random.normal`, `jax.random.multivariate_normal`,
`jax.random.truncated_normal` and `jax.random.lognormal`.

## Usage

``` r
random_normal(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_multivariate_normal(
  key,
  mean,
  cov,
  shape = NULL,
  dtype = NULL,
  method = "cholesky",
  out_sharding = NULL
)

random_truncated_normal(
  key,
  lower,
  upper,
  shape = NULL,
  dtype = NULL,
  out_sharding = NULL
)

random_lognormal(
  key,
  sigma = 1,
  shape = NULL,
  dtype = NULL,
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

  Integer vector, the result shape.

- dtype:

  A float dtype (`normal` also accepts complex dtypes).

- out_sharding:

  Ignored (one device).

- mean:

  Mean vector of shape `(..., n)`.

- cov:

  Positive definite covariance of shape `(..., n, n)`.

- method:

  `"cholesky"`, `"svd"` or `"eigh"`: factorization of `cov`.

- lower, upper:

  Truncation bounds, broadcast-compatible with each other.

- sigma:

  Standard deviation of the underlying normal distribution.

## Value

An array.
