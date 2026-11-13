# Stateful random number generator

Ports of `jax.experimental.random.stateful_rng` and
`jax.experimental.random.StatefulPRNG`: a generator in the style of
`numpy.random.default_rng` whose state updates implicitly. It is a
convenience wrapper around the explicit keys of
[`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md):
the generator holds a key and a counter stored in a reference (see
[`new_ref()`](https://adamrsophie.github.io/jaxR/reference/refs.md));
each draw uses `random_fold_in(key, counter)` and adds one to the
counter.

## Usage

``` r
experimental_random_stateful_rng(seed = NULL, impl = NULL)

experimental_random_StatefulPRNG(base_key, counter)
```

## Arguments

- seed:

  Integer seed. May be `NULL` at the top level only; the seed is then
  drawn from R's random number generator (JAX takes it from the
  operating system), so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) makes it
  reproducible.

- impl:

  Optional name of the generator implementation (for example
  `"threefry2x32"`).

- base_key:

  A typed scalar key from
  [`random_key()`](https://adamrsophie.github.io/jaxR/reference/random_key.md).

- counter:

  A reference holding a scalar integer.

## Value

A generator (class `jax_stateful_prng`).

## Details

The generator works at the top level and inside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) and
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md), as an
argument or as a closed over value. Like any reference it cannot be
returned from a function transformed by
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md), so it
cannot be a carry of
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)
or
[`lax_while_loop()`](https://adamrsophie.github.io/jaxR/reference/lax_while_loop.md).

Methods of the generator `rng` (sizes are an integer, an integer vector
or `NULL` for the broadcast shape of the parameters; `dtype = NULL` is
the default float or integer dtype):

- `rng$key(shape = integer(0))`: a new key, or an array of keys of that
  shape.

- `rng$random(size = NULL, dtype = NULL)`: floats in `[0, 1)`.

- `rng$uniform(low = 0, high = 1, size = NULL, dtype = NULL)`.

- `rng$normal(loc = 0, scale = 1, size = NULL, dtype = NULL)`.

- `rng$integers(low, high = NULL, size = NULL, dtype = NULL)`: integers
  in `[low, high)`, or in `[0, low)` when `high` is `NULL`.

- `rng$split(num)`: one generator holding `num` independent children, to
  pass to
  [`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md).

- `rng$spawn(n_children)`: a list of independent generators.

`rng$base_key` and `rng$counter` give the key and the counter reference
(the fields `_base_key` and `_counter` of JAX).

## Examples

``` r
if (FALSE) { # \dontrun{
rng <- experimental_random_stateful_rng(42)
rng$uniform()
rng$normal(size = 3L)
jit(function(x) x + rng$uniform())(1)
vmap(function(r, x) x + r$uniform())(rng$split(3L), jnp_zeros(3L))
} # }
```
