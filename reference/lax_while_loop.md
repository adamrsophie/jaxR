# While loop

Port of `jax.lax.while_loop`. Reverse mode differentiation is not
supported (as in JAX); forward mode and vmap are.

## Usage

``` r
lax_while_loop(cond_fun, body_fun, init_val)
```

## Arguments

- cond_fun:

  Function of the carry returning a scalar boolean.

- body_fun:

  Function of the carry returning a new carry of the same type.

- init_val:

  Initial carry (a pytree).
