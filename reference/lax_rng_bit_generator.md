# Stateless random bit generator

Port of `jax.lax.rng_bit_generator`: uniformly distributed random bits
from the XLA `RngBitGenerator` operation. The output is not guaranteed
to be stable across backends or compiler versions; most users should use
the `random_*` functions.

## Usage

``` r
lax_rng_bit_generator(
  key,
  shape,
  dtype = "uint32",
  algorithm = lax_RandomAlgorithm[["RNG_DEFAULT"]],
  out_sharding = NULL
)

lax_rng_bit_generator_p
```

## Arguments

- key:

  A `uint32[4]` (or `uint64[2]`) key.

- shape:

  Integer vector, the shape of the bits.

- dtype:

  An unsigned integer dtype.

- algorithm:

  One of
  [lax_RandomAlgorithm](https://adamrsophie.github.io/jaxR/reference/lax_RandomAlgorithm.md).

- out_sharding:

  Ignored (one device).

## Value

`list(new_key, bits)`.
