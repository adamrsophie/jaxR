# Trim leading and trailing zeros

Port of `jnp.trim_zeros`. The input must be concrete (not traced).

## Usage

``` r
jnp_trim_zeros(filt, trim = "fb", axis = NULL)
```

## Arguments

- filt:

  Array.

- trim:

  `"f"`, `"b"` or `"fb"` (front, back or both).

- axis:

  Axes to trim (default all).

## Value

An array.
