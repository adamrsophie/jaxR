# NumPy style indexing of jax arrays

`x[...]` follows NumPy semantics with 0-based indices (port of
`jax.numpy.ndarray.__getitem__`). Give one argument per dimension;
trailing dimensions that are not indexed are kept in full, and a missing
argument (`x[, 1]`) is a full slice. Index arguments can be:

- an R numeric scalar or a rank 0 integer jax value: selects one index
  and drops the dimension (negative values count from the end);

- [`sl()`](https://adamrsophie.github.io/jaxR/reference/sl.md): a
  half-open slice with optional step;

- an integer vector or array (R or jax): advanced indexing, with NumPy
  broadcasting and placement rules;

- a logical vector or a boolean array: a mask (concrete arrays only);

- [newaxis](https://adamrsophie.github.io/jaxR/reference/sl.md) or
  [ellipsis](https://adamrsophie.github.io/jaxR/reference/sl.md).

## Usage

``` r
# S3 method for class 'jax_value'
x[...]

# S3 method for class 'jax_value'
x[...] <- value
```

## Arguments

- x:

  A jax array or tracer.

- ...:

  Index arguments.

- value:

  Replacement values, broadcast to the indexed shape.

## Value

The indexed array, or the updated array.

## Details

Out of bounds indices are clamped when reading (use
[`at_get()`](https://adamrsophie.github.io/jaxR/reference/at.md) with
`mode = "fill"` for a fill value) and dropped when writing.
`x[...] <- value` returns an updated copy, as
[`at_set()`](https://adamrsophie.github.io/jaxR/reference/at.md) does;
arrays are immutable.
