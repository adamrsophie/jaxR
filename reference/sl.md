# Index helpers

`sl()` builds a Python style half-open slice for use inside `x[...]`:
`sl(1, 4)` is `1:4` in NumPy, `sl(2)` is `2:`, `sl(stop = 3)` is `:3`
and `sl(step = -1)` is `::-1`. Negative bounds count from the end.
`newaxis` inserts a dimension of size one (`None` in NumPy) and
`ellipsis` stands for as many full slices as needed (`...` in NumPy).

## Usage

``` r
sl(start = NULL, stop = NULL, step = NULL)

newaxis

ellipsis
```

## Arguments

- start:

  First index (inclusive); `NULL` for the beginning.

- stop:

  Last index (exclusive); `NULL` for the end.

- step:

  Step; `NULL` for 1. May be negative.

## Value

An index object.
