# Ragged all to all collective

Port of `jax.lax.ragged_all_to_all`. As in JAX, it needs a multi device
mapped context (shard_map); on one device it raises an error, and under
`vmap` over its own axis it raises `NotImplementedError` like JAX's
batching rule.

## Usage

``` r
lax_ragged_all_to_all(
  operand,
  output,
  input_offsets,
  send_sizes,
  output_offsets,
  recv_sizes,
  axis_name,
  axis_index_groups = NULL
)
```

## Arguments

- operand, output:

  Data to send and the array receiving it.

- input_offsets, send_sizes, output_offsets, recv_sizes:

  1D integer arrays.

- axis_name:

  Axis name(s).

- axis_index_groups:

  Not supported.

## Value

An array shaped like `output`.
