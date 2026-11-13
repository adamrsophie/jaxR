# Jacobian-vector product

Port of `jax.jvp`.

## Usage

``` r
jvp(fun, primals, tangents, has_aux = FALSE)
```

## Arguments

- fun:

  Function of pytrees.

- primals:

  List of primal arguments.

- tangents:

  List of tangents with the same structure as `primals`. Each tangent
  has the shape and dtype of its primal; the tangent of an integer or
  boolean primal is a float0 array (`jnp_zeros(shape, "float0")`).

- has_aux:

  Whether `fun` returns `list(output, aux)`.

## Value

`list(primals_out, tangents_out)` (plus aux).
