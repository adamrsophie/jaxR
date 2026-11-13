# Copy arrays to the host

Port of `jax.copy_to_host_async`. jaxR copies synchronously: the host
copies of the array leaves are fetched (and cached) now.

## Usage

``` r
copy_to_host_async(x)
```

## Arguments

- x:

  A pytree.

## Value

`x`, invisibly.
