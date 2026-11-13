# Convert a padding specification to explicit padding pairs

Port of `jax.lax.padtype_to_pads`.

## Usage

``` r
lax_padtype_to_pads(in_shape, window_shape, window_strides, padding)
```

## Arguments

- in_shape:

  Integer vector, the input spatial shape.

- window_shape:

  Integer vector, the window (kernel) spatial shape.

- window_strides:

  Integer vector, the spatial strides.

- padding:

  One of `"VALID"`, `"SAME"` or `"SAME_LOWER"` (case insensitive).

## Value

A list with one `c(low, high)` integer pair per spatial dimension.
