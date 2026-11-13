# General dot product

Port of `jax.lax.dot_general`.

## Usage

``` r
lax_dot_general(
  lhs,
  rhs,
  dimension_numbers,
  precision = NULL,
  preferred_element_type = NULL
)
```

## Arguments

- lhs, rhs:

  Arrays.

- dimension_numbers:

  `list(list(lhs_contracting, rhs_contracting), list(lhs_batch, rhs_batch))`
  with 0-based dimensions.

- precision:

  Optional precision (`"default"`, `"high"`, `"highest"`).

- preferred_element_type:

  Optional result dtype.
