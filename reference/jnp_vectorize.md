# Vectorize a function with broadcasting

Port of `jnp.vectorize`: the function is mapped with
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md) over
the broadcast leading dimensions of its arguments. With a generalized
ufunc `signature` such as `"(n,m),(m)->(n)"` the trailing core
dimensions are passed whole.

## Usage

``` r
jnp_vectorize(pyfunc, excluded = integer(0), signature = NULL)
```

## Arguments

- pyfunc:

  Function to vectorize.

- excluded:

  Positions (0-based) or names of arguments passed through unchanged.
  Named arguments that are not excluded are ignored, as upstream.

- signature:

  Optional generalized ufunc signature.

## Value

The vectorized function. Multiple outputs are returned (and must be
returned by `pyfunc`) as an unnamed list.
