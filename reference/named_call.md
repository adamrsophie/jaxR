# Name a function or a scope of staged computations

Ports of `jax.named_call` and `jax.named_scope`. JAX records these names
in the name stack of staged operations (they appear in profiler traces
and HLO metadata); jaxR keeps no name stack, so `named_call()` returns
`fun` and `named_scope()` evaluates `code`. Both are accepted so that
JAX code ports unchanged.

## Usage

``` r
named_call(fun, name = NULL)

named_scope(name, code)
```

## Arguments

- fun:

  A function.

- name:

  A name (`named_call()`: optional).

- code:

  Expression to evaluate.

## Value

`named_call()`: `fun`; `named_scope()`: the value of `code`.
