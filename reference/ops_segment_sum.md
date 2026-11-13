# Segment reductions

Ports of `jax.ops.segment_sum`, `segment_prod`, `segment_max` and
`segment_min`: reduce the entries of `data` (along its first axis) that
share a segment id. Segments without entries hold the identity of the
reduction. Ids outside `[0, num_segments)` are dropped (negative ids do
not wrap).

## Usage

``` r
ops_segment_sum(
  data,
  segment_ids,
  num_segments = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  bucket_size = NULL,
  mode = NULL
)

ops_segment_prod(
  data,
  segment_ids,
  num_segments = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  bucket_size = NULL,
  mode = NULL
)

ops_segment_max(
  data,
  segment_ids,
  num_segments = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  bucket_size = NULL,
  mode = NULL
)

ops_segment_min(
  data,
  segment_ids,
  num_segments = NULL,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  bucket_size = NULL,
  mode = NULL
)
```

## Arguments

- data:

  Array.

- segment_ids:

  Integer array of 0-based segment ids, one per entry of the first axis
  of `data`.

- num_segments:

  Number of segments. `NULL` uses `max(segment_ids) + 1`, which requires
  concrete ids (so it cannot be used under
  [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md)).

- indices_are_sorted:

  Whether `segment_ids` is known to be sorted.

- unique_indices:

  Whether `segment_ids` is known to be free of duplicates.

- bucket_size:

  Optional size of the buckets used to group the reduction.

- mode:

  Out of bounds mode (see
  [lax_GatherScatterMode](https://adamrsophie.github.io/jaxR/reference/lax_GatherScatterMode.md));
  the default drops.

## Value

An array with leading dimension `num_segments`.
