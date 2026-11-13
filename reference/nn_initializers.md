# Neural network initializers

Ports of `jax.nn.initializers`. `nn_initializers_zeros` and
`nn_initializers_ones` are initializers; the other functions build
initializers. An initializer is a function
`init(key, shape, dtype = NULL, out_sharding = NULL)` (the `Initializer`
protocol of JAX) returning an array of the given shape and dtype
(default float). Random initializers give the same values as JAX for the
same key.

## Usage

``` r
nn_initializers_zeros(key, shape, dtype = NULL, out_sharding = NULL)

nn_initializers_ones(key, shape, dtype = NULL, out_sharding = NULL)

nn_initializers_constant(value, dtype = NULL)

nn_initializers_uniform(scale = 0.01, dtype = NULL)

nn_initializers_normal(stddev = 0.01, dtype = NULL)

nn_initializers_truncated_normal(
  stddev = 0.01,
  dtype = NULL,
  lower = -2,
  upper = 2
)

nn_initializers_variance_scaling(
  scale,
  mode,
  distribution,
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_glorot_uniform(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_xavier_uniform(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_glorot_normal(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_xavier_normal(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_lecun_uniform(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_lecun_normal(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_he_uniform(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_kaiming_uniform(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_he_normal(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_kaiming_normal(
  in_axis = -2L,
  out_axis = -1L,
  batch_axis = integer(0),
  dtype = NULL
)

nn_initializers_orthogonal(scale = 1, column_axis = -1L, dtype = NULL)

nn_initializers_delta_orthogonal(scale = 1, column_axis = -1L, dtype = NULL)
```

## Arguments

- key:

  A PRNG key
  ([`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
  or
  [`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md));
  ignored by the constant initializers.

- shape:

  Integer vector.

- dtype:

  Dtype (default float).

- out_sharding:

  Accepted for compatibility (jaxR runs on one device); the orthogonal
  initializers require `NULL`.

- value:

  Fill value.

- scale:

  Upper bound of `uniform`, scaling factor of `variance_scaling` and
  `orthogonal`.

- stddev:

  Standard deviation.

- lower, upper:

  Truncation bounds of `truncated_normal` (in units of `stddev`).

- mode:

  One of `"fan_in"`, `"fan_out"`, `"fan_avg"` and `"fan_geo_avg"`.

- distribution:

  One of `"truncated_normal"`, `"normal"` and `"uniform"`.

- in_axis, out_axis, batch_axis:

  0-based axis or axes of the input dimension, the output dimension and
  the ignored (batch) dimensions of the weights.

- column_axis:

  0-based axis holding the columns that should be orthogonal.

## Value

An array (`zeros`, `ones`) or an initializer function.

## Details

`nn_initializers_variance_scaling` draws from a (truncated) normal
distribution with standard deviation `sqrt(scale / n)` or a uniform
distribution with that standard deviation, where `n` is the number of
inputs (`"fan_in"`), outputs (`"fan_out"`), their mean (`"fan_avg"`) or
their geometric mean (`"fan_geo_avg"`). Axes that are not input, output
or batch axes form the receptive field of a convolution kernel. Complex
dtypes draw from a disk (`"uniform"`) or a truncated complex normal
distribution. The Glorot (Xavier), LeCun and He (Kaiming) initializers
are its specializations.

## Examples

``` r
if (FALSE) { # \dontrun{
init <- nn_initializers_glorot_normal()
init(random_key(42), c(2, 3))
nn_initializers_zeros(random_key(0), c(2, 2))
} # }
```
