# Gradient checkpointing

Port of `jax.checkpoint` (alias `remat`): the function's intermediate
values are not saved for the backward pass but recomputed, except those
allowed by `policy`.

## Usage

``` r
checkpoint(fun, prevent_cse = TRUE, policy = NULL, static_argnums = integer(0))

remat(fun, prevent_cse = TRUE, policy = NULL, static_argnums = integer(0))
```

## Arguments

- fun:

  Function.

- prevent_cse:

  Keep XLA from merging the recomputation with the forward pass.

- policy:

  A policy from
  [checkpoint_policies](https://adamrsophie.github.io/jaxR/reference/checkpoint_policies.md)
  or `NULL` (save nothing).

- static_argnums:

  0-based positions of static arguments.

## Value

The wrapped function.
