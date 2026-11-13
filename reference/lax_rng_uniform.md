# Stateful uniform random numbers

Port of `jax.lax.rng_uniform`: uniform numbers in `[a, b)` from the XLA
`Rng` operation. Experimental and nondeterministic; use
[`random_uniform()`](https://adamrsophie.github.io/jaxR/reference/random_uniform.md)
instead.

## Usage

``` r
lax_rng_uniform(a, b, shape)

lax_rng_uniform_p
```

## Arguments

- a, b:

  Scalars of the same dtype.

- shape:

  Integer vector.

## Value

An array of shape `shape`.
