# Name an intermediate value for checkpoint policies

Port of `jax.ad_checkpoint.checkpoint_name`.

## Usage

``` r
checkpoint_name(x, name)
```

## Arguments

- x:

  A pytree of arrays.

- name:

  Name used by
  [checkpoint_policies](https://adamrsophie.github.io/jaxR/reference/checkpoint_policies.md)
  such as `save_only_these_names`.
