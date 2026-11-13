# General reduction

Port of `jax.lax.reduce`.

## Usage

``` r
lax_reduce(operands, init_values, computation, dimensions)
```

## Arguments

- operands:

  Array or list of arrays.

- init_values:

  Scalar or list of scalars.

- computation:

  Function of `(accumulators, values)`; with several operands it
  receives two lists and returns a list.

- dimensions:

  0-based dimensions to reduce.
