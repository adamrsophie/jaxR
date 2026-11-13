# Evenly spaced numbers over an interval

Port of `jnp.linspace`.

## Usage

``` r
jnp_linspace(
  start,
  stop,
  num = 50L,
  endpoint = TRUE,
  retstep = FALSE,
  dtype = NULL,
  axis = 0L
)
```

## Arguments

- start, stop:

  Interval ends (scalars or arrays that broadcast).

- num:

  Number of samples.

- endpoint:

  Include `stop`.

- retstep:

  Also return the step: the result is then `list(samples, step)`.

- dtype:

  Optional dtype (default: the inexact promoted type of the ends).

- axis:

  0-based axis of the samples in the result.
