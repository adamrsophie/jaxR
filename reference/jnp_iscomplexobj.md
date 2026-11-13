# Dtype and scalar queries

`jnp_iscomplexobj` and `jnp_isrealobj` port `jnp.iscomplexobj` and
`jnp.isrealobj` (test the dtype of a value), `jnp_isscalar` ports
`jnp.isscalar` (a length one R vector without `dim`, or a rank 0 array)
and `jnp_iterable` ports `numpy.iterable` (a list, an R vector that is
not a scalar, or an array of rank at least one).

## Usage

``` r
jnp_iscomplexobj(x)

jnp_isrealobj(x)

jnp_isscalar(element)

jnp_iterable(y)
```

## Arguments

- x, element, y:

  Values.

## Value

A logical scalar.
