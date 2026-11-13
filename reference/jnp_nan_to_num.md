# Replace NaN and infinities

Port of `jnp.nan_to_num`.

## Usage

``` r
jnp_nan_to_num(x, copy = TRUE, nan = 0, posinf = NULL, neginf = NULL)
```

## Arguments

- x:

  Array.

- copy:

  Ignored.

- nan:

  Replacement for NaN.

- posinf, neginf:

  Replacements for infinities (default: the largest and smallest finite
  values of the dtype).

## Value

An array.
