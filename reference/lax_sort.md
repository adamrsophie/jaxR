# Sort arrays

Port of `jax.lax.sort`. Floating point values follow a total order in
which -0.0 and 0.0 are equivalent and NaNs sort to the end; complex
values sort lexicographically on (real, imag).

## Usage

``` r
lax_sort(operand, dimension = -1L, is_stable = TRUE, num_keys = 1L)
```

## Arguments

- operand:

  Array or list of arrays of equal shapes.

- dimension:

  0-based dimension to sort along (negative counts from the end).

- is_stable:

  Use a stable sort.

- num_keys:

  Number of leading operands used as lexicographic sort keys.

## Value

The sorted array, or a list of arrays permuted like the keys.
