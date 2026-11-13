# Count occurrences of non-negative integers

Port of `jnp.bincount`. Negative values count toward bin 0 and values at
or above `length` are dropped. The output length depends on the data
unless `length` is given, which is required inside `jit` and other
transformations.

## Usage

``` r
jnp_bincount(x, weights = NULL, minlength = 0L, length = NULL)
```

## Arguments

- x:

  1-D integer (or boolean) array.

- weights:

  Optional weights with the shape of `x`.

- minlength:

  Minimum number of bins.

- length:

  Optional static number of bins.

## Value

An array of counts (the dtype of `weights`, default integer).
