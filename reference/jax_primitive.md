# Primitive objects

Every operation of jaxR is a primitive, as in JAX. Primitive objects
print as `<primitive name>` and expose `$name`, `$multiple_results` and
`$bind(...)`, the port of `Primitive.bind`: unnamed arguments are
operands and named arguments are parameters, so `lax_add_p$bind(1, 2)`
equals `lax_add(1, 2)` and
`lax_reshape_p$bind(x, new_sizes = c(2L, 3L))` reshapes `x`. The
exported objects carry the JAX names with the module prefix:
`jax.lax.add_p` is `lax_add_p`, `jax.lax.linalg.qr_p` is
`lax_linalg_qr_p`, `jax.custom_derivatives.custom_jvp_call_p` is
`custom_jvp_call_p`.

## Usage

``` r
# S3 method for class 'jax_primitive'
print(x, ...)
```

## Format

Primitive objects.

## Arguments

- x:

  A primitive.

- ...:

  Ignored.

## Examples

``` r
lax_add_p
#> <primitive add>
if (FALSE) { # \dontrun{
lax_add_p$bind(1, 2)
} # }
```
