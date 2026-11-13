# Select and apply one of several branches

Port of `jax.lax.switch`; `index` is 0-based and clamped to the valid
range.

## Usage

``` r
lax_switch(index, branches, ...)
```

## Arguments

- index:

  Scalar integer.

- branches:

  List of functions.

- ...:

  Operands.
