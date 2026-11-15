# Device memory layouts

Ports of `jax.experimental.layout.Layout`, `Layout.AUTO`,
`Layout.for_array`, `Layout.update` and `Format`. A layout lists the
dimensions of an array from the most major to the most minor (0 based):
`c(0, 1)` is the row major order of a matrix and `c(1, 0)` the column
major one. `experimental_layout_Layout_AUTO` lets the compiler choose. A
`Format` pairs a layout with a sharding; jaxR runs on one device, so the
sharding is `NULL` or a device name from
[`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md).

## Usage

``` r
experimental_layout_Layout(
  major_to_minor,
  tiling = NULL,
  sub_byte_element_size_in_bits = 0L
)

experimental_layout_Layout_AUTO

experimental_layout_Layout_for_array(x)

experimental_layout_Layout_update(x, ...)

experimental_layout_Format(layout = NULL, sharding = NULL)
```

## Arguments

- major_to_minor:

  Integer vector, a permutation of `0:(ndim - 1)`.

- tiling:

  `NULL` or a list of integer vectors (tiles). The CPU device has no
  tiled layouts, so only `NULL` and the empty list can be used in a
  computation.

- sub_byte_element_size_in_bits:

  Integer.

- x:

  An array (for `for_array`) or a layout (for `update`).

- ...:

  Fields to replace (`major_to_minor`, `tiling`,
  `sub_byte_element_size_in_bits`).

- layout:

  `NULL`, `experimental_layout_Layout_AUTO` or a layout.

- sharding:

  `NULL` or a device name.

## Value

A `jax_Layout` or `jax_Format` object.

## Examples

``` r
if (FALSE) { # \dontrun{
l <- experimental_layout_Layout(c(1L, 0L))
l
experimental_layout_Layout_for_array(jnp_ones(c(2L, 3L)))
experimental_layout_Format(l, devices()[1])
} # }
```
