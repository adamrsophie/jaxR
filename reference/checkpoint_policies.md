# Checkpoint policies

Port of `jax.checkpoint_policies`. A policy is a function
`(prim_name, params)` returning `TRUE` when the output of that primitive
may be saved as a residual instead of being recomputed in the backward
pass. Offloading policies behave like the corresponding saving policies
(there is one memory space).

## Usage

``` r
checkpoint_policies
```

## Format

A list of policies and policy constructors.
