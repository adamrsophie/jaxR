# Random permutations and choices

Ports of `jax.random.permutation` and `jax.random.choice`. Axes are
0-based.

## Usage

``` r
random_permutation(key, x, axis = 0L, independent = FALSE, out_sharding = NULL)

random_choice(
  key,
  a,
  shape = integer(0),
  replace = TRUE,
  p = NULL,
  axis = 0L,
  mode = NULL
)
```

## Arguments

- key:

  A PRNG key from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- x:

  An integer `n` (permute `0:(n - 1)`) or an array.

- axis:

  Axis along which to shuffle or select.

- independent:

  Whether each slice along `axis` is shuffled independently.

- out_sharding:

  Ignored (one device).

- a:

  An integer `n` (choose from `0:(n - 1)`) or an array.

- shape:

  Integer vector, the result shape.

- replace:

  Whether to sample with replacement.

- p:

  Optional probabilities of the entries of `a` along `axis`.

- mode:

  `"low"` or `"high"`: bits used by the Gumbel sampler (`p` given,
  `replace = FALSE`).

## Value

An array.
