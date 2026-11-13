# Out of bounds modes for gather and scatter

Port of `jax.lax.GatherScatterMode`. `CLIP` clamps indices so that the
whole window is in range. `FILL_OR_DROP` fills gathered windows that are
out of bounds with a constant and drops scattered windows that are out
of bounds. `PROMISE_IN_BOUNDS` performs no checking (with XLA, gathers
clamp and scatters drop). `from_any()` parses `"clip"`, `"fill"`,
`"drop"`, `"promise_in_bounds"`, `"one_hot"` or `NULL` (fill or drop).

## Usage

``` r
lax_GatherScatterMode
```
