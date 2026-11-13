# Index expression builders

Ports of `jnp.s_` and `jnp.index_exp` (NumPy's): `jnp_s_[...]` returns
the index written between the brackets (one index as is, several as an
index tuple that `x[...]` accepts), and `jnp_index_exp[...]` always
returns an index tuple. Indices are 0-based and missing ones are full
slices, as in `x[...]`.

## Usage

``` r
jnp_s_

jnp_index_exp
```

## Format

Index builder objects.

## Examples

``` r
if (FALSE) { # \dontrun{
x <- jnp_reshape(jnp_arange(12), c(3L, 4L))
x[jnp_s_[1L, sl(0L, 2L)]]
} # }
```
