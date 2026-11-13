# One-dimensional linear interpolation

Port of `jnp.interp` for increasing `xp`.

## Usage

``` r
jnp_interp(x, xp, fp, left = NULL, right = NULL, period = NULL)
```

## Arguments

- x:

  Points to evaluate.

- xp, fp:

  1-D arrays of coordinates and values.

- left, right:

  Values outside the range (default `fp[0]` and `fp[-1]`), or
  `"extrapolate"`.

- period:

  Optional period of `xp`.

## Value

An array.
