# Ahead of time tracing, lowering and compilation

Port of `jax.stages`. A function made by
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) has the
methods of JAX's `Wrapped` protocol, reached with `$`:

## Usage

``` r
# S3 method for class 'jax_jitted'
x$name

# S3 method for class 'jax_stage'
x$name

# S3 method for class 'jax_stage'
print(x, ...)

# S3 method for class 'jax_jitted'
print(x, ...)
```

## Arguments

- x:

  A function made by
  [`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md), or a
  stage.

- name:

  Field or method name.

- ...:

  Ignored.

## Value

The field, or a function for a method.

## Details

- `f$trace(...)` traces `f` for the given arguments and returns a
  *Traced* stage.

- `f$lower(...)` is `f$trace(...)$lower()`: a *Lowered* stage.

- `f$eval_shape(...)` returns the shapes and dtypes of the result
  ([`eval_shape()`](https://adamrsophie.github.io/jaxR/reference/eval_shape.md)).

- `f$clear_cache()` drops the compiled versions of `f`.

The arguments can be arrays or
[`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
values. Each stage is read with `$` as well:

|  |  |
|----|----|
| Stage | Fields and methods |
| Traced | `jaxpr`, `fun_name`, `lower()`; a Traced stage is a function and can be called |
| Lowered | `as_text(dialect = NULL)`, `compiler_ir(dialect = NULL)`, `compile()`, `cost_analysis()` |
| Compiled | `as_text()`, `cost_analysis()`, `memory_analysis()`, `runtime_executable()`, `call(...)`; a Compiled stage is a function and can be called |
| all | `args_info`, `out_info`, `in_tree`, `out_tree`, `in_avals` |

`as_text()` of a Lowered stage is the StableHLO module (the only
dialect). The analyses return `NULL`, as they do in JAX when the backend
provides none. A Compiled stage takes the non static arguments of `f`,
accepts only the shapes and dtypes it was compiled for, and cannot be
transformed again.

## Examples

``` r
if (FALSE) { # \dontrun{
f <- jit(function(x) jnp_sum(x^2))
lowered <- f$lower(as_jax(c(1, 2, 3)))
cat(lowered$as_text())
compiled <- lowered$compile()
compiled(as_jax(c(1, 2, 3)))
} # }
```
