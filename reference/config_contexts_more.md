# Further configuration contexts

Ports of `jax.enable_checks`, `jax.debug_key_reuse`,
`jax.legacy_prng_key`, `jax.enable_custom_prng`,
`jax.softmax_custom_jvp`, `jax.allow_f16_reductions`,
`jax.array_garbage_collection_guard` and `jax.thread_guard`. Each
evaluates `code` with the option set. `legacy_prng_key("warn")` or
`("error")` makes
[`random_PRNGKey()`](https://adamrsophie.github.io/jaxR/reference/random_key.md)
warn or fail; `softmax_custom_jvp` selects the custom JVP of
[`nn_softmax()`](https://adamrsophie.github.io/jaxR/reference/nn_softmax.md).
The other options are accepted for compatibility: jaxR has one thread,
frees device buffers with R's garbage collector, keeps its checks always
on, and does not track key reuse.

## Usage

``` r
enable_checks(new_val = TRUE, code)

debug_key_reuse(new_val = TRUE, code)

legacy_prng_key(new_val = TRUE, code)

enable_custom_prng(new_val = TRUE, code)

softmax_custom_jvp(new_val = TRUE, code)

allow_f16_reductions(new_val = TRUE, code)

array_garbage_collection_guard(new_val = TRUE, code)

thread_guard(new_val = TRUE, code)
```

## Arguments

- new_val:

  New value of the option.

- code:

  Expression evaluated with the option set.
