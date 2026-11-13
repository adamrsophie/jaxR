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
`debug_key_reuse(TRUE, code)` checks that no typed key is used twice
(port of `jax.experimental.key_reuse`): a key used by a random function
is consumed, and using a consumed key raises an error of class
`jax_key_reuse_error`, also inside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`lax_cond()`](https://adamrsophie.github.io/jaxR/reference/lax_cond.md),
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md),
[`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md)
and [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md).
The other options are accepted for compatibility: jaxR has one thread,
frees device buffers with R's garbage collector and keeps its checks
always on.

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
