# Discrete random values

Ports of `jax.random.bernoulli`, `rademacher`, `categorical`,
`geometric`, `poisson`, `binomial` and `multinomial`. Axes are 0-based.

## Usage

``` r
random_bernoulli(key, p = 0.5, shape = NULL, mode = "low", out_sharding = NULL)

random_rademacher(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_categorical(
  key,
  logits,
  axis = -1L,
  shape = NULL,
  replace = TRUE,
  mode = NULL,
  out_sharding = NULL
)

random_geometric(key, p, shape = NULL, dtype = NULL, out_sharding = NULL)

random_poisson(
  key,
  lam,
  shape = NULL,
  dtype = NULL,
  method = "exact",
  out_sharding = NULL
)

random_binomial(key, n, p, shape = NULL, dtype = NULL)

random_multinomial(key, n, p, shape = NULL, dtype = NULL, unroll = 1L)
```

## Arguments

- key:

  A PRNG key from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- p:

  Probability (of success, or of each outcome for `multinomial`).

- shape:

  Integer vector; `NULL` (where allowed) broadcasts the parameters.

- mode:

  `"low"` or `"high"` (also `"highest"` for the Gumbel sampler of
  `categorical`): how many random bits to use.

- out_sharding:

  Ignored (one device).

- dtype:

  Result dtype.

- logits:

  Unnormalized log probabilities.

- axis:

  Axis of `logits` holding the categories.

- replace:

  Whether `categorical` samples with replacement.

- lam:

  Poisson rate.

- method:

  `"exact"` or `"approximate"`.

- n:

  Number of trials.

- unroll:

  Unroll parameter of the scan in `multinomial`.

## Value

An array.
