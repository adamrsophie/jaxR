# Parallel prefix scan with an associative operation

Port of `jax.lax.associative_scan`.

## Usage

``` r
lax_associative_scan(fn, elems, reverse = FALSE, axis = 0L)
```

## Arguments

- fn:

  Associative function `fn(a, b)` applied elementwise over `axis`; its
  arguments and result have the structure of `elems`.

- elems:

  Array or pytree (list) of arrays with equal sizes along `axis`.

- reverse:

  Scan from the end.

- axis:

  0-based axis.

## Value

Arrays with the structure of `elems`.
