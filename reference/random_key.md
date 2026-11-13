# PRNG keys

Ports of the key functions of `jax.random`. `random_key()` returns a
typed key array (a scalar array of keys whose dtype names the PRNG
implementation, `key<fry>` for threefry2x32); `random_PRNGKey()` returns
a legacy raw key, a `uint32` array of shape `2`. Both are accepted by
all samplers. Typed key arrays are pytrees: they pass through
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) and
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) (with
float0 tangents), can be indexed with `[` and split into a list with
[`as.list()`](https://rdrr.io/r/base/list.html).

## Usage

``` r
random_key(seed, impl = NULL, dtype = NULL)

random_PRNGKey(seed, impl = NULL)

random_fold_in(key, data)

random_split(key, num = 2L)

random_key_impl(keys)

random_key_data(keys)

random_wrap_key_data(key_bits_array, impl = NULL, dtype = NULL)

random_key_dtype(impl_spec = NULL)

random_clone(key)

random_Key(x)
```

## Arguments

- seed:

  An integer scalar (an R whole number or an integer jax scalar). R
  numbers are treated as Python ints: they wrap to 32 bits unless x64 is
  enabled.

- impl:

  Optional PRNG implementation name.

- dtype:

  Optional key dtype (as returned by `random_key_dtype()`) or
  implementation name; an alternative to `impl`.

- key, keys:

  A key (typed or raw) or key array.

- data:

  A 32-bit integer scalar to fold into the key.

- num:

  Number of keys (or shape of the key array) to produce.

- key_bits_array:

  A `uint32` array whose trailing dimensions are the key shape of the
  implementation.

- impl_spec:

  Optional implementation name.

- x:

  A value (returned unchanged by `random_Key()`).

## Value

`random_key()`, `random_split()`, `random_fold_in()`, `random_clone()`
and `random_wrap_key_data()` return keys (raw keys stay raw);
`random_key_data()` the `uint32` key data; `random_key_impl()` the
implementation name; `random_key_dtype()` the key dtype name.

## Details

The implementations are `"threefry2x32"` (the default, see the option
`jax_default_prng_impl`), `"rbg"` and `"unsafe_rbg"`. The option
`jax_threefry_partitionable` (default `TRUE`) selects the threefry split
and bits algorithms, as in JAX.
