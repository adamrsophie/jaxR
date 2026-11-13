# Numbers spaced evenly on a log scale

Ports of `jnp.logspace` (`base^linspace(start, stop)`) and
`jnp.geomspace` (a geometric progression from `start` to `stop`).

## Usage

``` r
jnp_logspace(
  start,
  stop,
  num = 50L,
  endpoint = TRUE,
  base = 10,
  dtype = NULL,
  axis = 0L
)

jnp_geomspace(start, stop, num = 50L, endpoint = TRUE, dtype = NULL, axis = 0L)
```

## Arguments

- start, stop:

  Interval ends (exponents for `jnp_logspace`).

- num:

  Number of samples.

- endpoint:

  Include `stop`.

- base:

  Base of the log scale.

- dtype:

  Optional dtype.

- axis:

  0-based axis of the samples in the result.

## Value

An array.
