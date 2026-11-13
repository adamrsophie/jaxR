# Named axis collectives

Ports of `jax.lax.psum`, `pmean`, `pmax`, `pmin`, `axis_index`,
`all_gather`, `ppermute`, `pshuffle`, `pbroadcast`, `all_to_all`,
`pswapaxes` and `psum_scatter`. On one device they apply to axes named
by an enclosing `vmap(..., axis_name = name)`.

## Usage

``` r
lax_psum(x, axis_name, axis_index_groups = NULL)

lax_pmax(x, axis_name, axis_index_groups = NULL)

lax_pmin(x, axis_name, axis_index_groups = NULL)

lax_pmean(x, axis_name, axis_index_groups = NULL)

lax_axis_index(axis_name)

lax_axis_size(axis_name)

lax_all_gather(
  x,
  axis_name,
  axis_index_groups = NULL,
  axis = 0L,
  tiled = FALSE
)

lax_psum_scatter(
  x,
  axis_name,
  scatter_dimension = 0L,
  axis_index_groups = NULL,
  tiled = FALSE
)

lax_psum_scatter_start(
  x,
  axis_name,
  scatter_dimension = 0L,
  axis_index_groups = NULL,
  tiled = FALSE
)

lax_ppermute(x, axis_name, perm)

lax_pshuffle(x, axis_name, perm)

lax_pbroadcast(x, axis_name, source)

lax_all_to_all(
  x,
  axis_name,
  split_axis,
  concat_axis,
  axis_index_groups = NULL,
  tiled = FALSE
)

lax_pswapaxes(x, axis_name, axis, axis_index_groups = NULL)
```

## Arguments

- x:

  A pytree of arrays.

- axis_name:

  Axis name (or several names for reductions).

- axis_index_groups:

  Not supported (must be `NULL`).

- axis:

  0-based position of the gathered axis.

- tiled:

  Concatenate instead of stacking.

- scatter_dimension:

  0-based dimension split across the axis.

- perm:

  List of `c(source, destination)` index pairs (0-based).

- source:

  0-based source index.

- split_axis, concat_axis:

  0-based axes.

## Value

`lax_psum_scatter_start()` returns a future: call `$done()` on it to get
the result of `lax_psum_scatter()`. With one device the reduction runs
eagerly, so there is no communication to overlap.
