# Uniform random bits, floats and integers

Ports of `jax.random.bits`, `jax.random.uniform` and
`jax.random.randint`. For a given key the results equal those of JAX bit
for bit.

## Usage

``` r
random_bits(key, shape = integer(0), dtype = NULL, out_sharding = NULL)

random_uniform(
  key,
  shape = integer(0),
  dtype = NULL,
  minval = 0,
  maxval = 1,
  out_sharding = NULL
)

random_randint(key, shape, minval, maxval, dtype = NULL, out_sharding = NULL)
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

  Result dtype (`bits`: an unsigned integer dtype, default `uint32`, or
  `uint64` with x64; `uniform`: a float dtype; `randint`: an integer
  dtype).

- out_sharding:

  Ignored (one device).

- minval, maxval:

  Bounds of the range `[minval, maxval)`, broadcast-compatible with
  `shape`.

## Value

An array of shape `shape`.
