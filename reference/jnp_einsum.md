# Einstein summation

Ports of `jnp.einsum` and `jnp.einsum_path`. Subscripts follow NumPy
(`"ij,jk->ik"`, implicit output, `...` broadcasting); the interleaved
form `jnp_einsum(x, c(0, 1), y, c(1, 2), c(0, 2))` is also accepted
(sublists may be lists containing
[ellipsis](https://adamrsophie.github.io/jaxR/reference/sl.md)).
Contractions are planned pairwise as in `opt_einsum`:
`optimize = "auto"` (optimal for up to four operands, greedy beyond),
`TRUE` (optimal), `"optimal"`, `"greedy"`, `FALSE` (left to right) or an
explicit path (a list of 0-based index pairs).

## Usage

``` r
jnp_einsum(
  subscripts,
  ...,
  out = NULL,
  optimize = "auto",
  precision = NULL,
  preferred_element_type = NULL
)

jnp_einsum_path(subscripts, ..., optimize = "auto")
```

## Arguments

- subscripts:

  Subscript string (or the first operand of the interleaved form).

- ...:

  Operands (and sublists in the interleaved form).

- out:

  Not supported.

- optimize:

  Path optimization (see Details).

- precision:

  Optional precision for the dot products.

- preferred_element_type:

  Optional accumulation and result dtype.

## Value

`jnp_einsum` returns an array. `jnp_einsum_path` returns a list of two
elements: the path (a list of 0-based integer vectors) and a printable
summary string.
