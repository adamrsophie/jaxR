# Pack and unpack bits

Ports of `jnp.packbits` and `jnp.unpackbits` (uint8 results).

## Usage

``` r
jnp_packbits(a, axis = NULL, bitorder = "big")

jnp_unpackbits(a, axis = NULL, count = NULL, bitorder = "big")
```

## Arguments

- a:

  Integer or boolean array (`jnp_unpackbits` requires uint8).

- axis:

  Axis (0-based) or `NULL` for the flattened array.

- bitorder:

  `"big"` or `"little"`.

- count:

  Number of bits to keep (padding with zeros if larger).

## Value

A uint8 array.
