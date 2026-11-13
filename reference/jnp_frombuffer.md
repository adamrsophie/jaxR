# Arrays from raw bytes, text, functions or iterators

`jnp_frombuffer` ports `jnp.frombuffer` for an R raw vector (little
endian, as NumPy on common platforms); `jnp_fromstring` ports
`jnp.fromstring` (text mode, `sep` is required; whitespace around
separators is ignored). `jnp_fromfunction` ports `jnp.fromfunction`:
`function` is called once with index arrays built by nested `vmap`
(further arguments in `...` are passed unmapped). `jnp_fromfile` and
`jnp_fromiter` raise errors, as in JAX.

## Usage

``` r
jnp_frombuffer(buffer, dtype = "float", count = -1L, offset = 0L)

jnp_fromstring(string, dtype = "float", count = -1L, sep)

jnp_fromfunction(`function`, shape, ..., dtype = "float")

jnp_fromfile(...)

jnp_fromiter(...)
```

## Arguments

- buffer:

  Raw vector.

- dtype:

  Dtype of the data (`"float"` is float64 before canonicalization).

- count:

  Number of items (`-1` for all).

- offset:

  Number of bytes to skip.

- string:

  Character scalar.

- sep:

  Separator.

- function:

  Function of the index arrays.

- shape:

  Integer vector.

- ...:

  Further arguments to `function`; ignored by `jnp_fromfile` and
  `jnp_fromiter`.

## Value

An array.
