# User contexts that affect jit caching

Port of `jax.make_user_context`: returns a function `ctx(value, code)`
that evaluates `code` with the context set to `value`. `jit` caches are
keyed on the values of all user contexts, so a function traced under one
value is traced again under another. `ctx()` with no arguments returns
the current value.

## Usage

``` r
make_user_context(default_value = NULL)
```

## Arguments

- default_value:

  The value outside any `ctx(value, code)` call.

## Value

A context function.
