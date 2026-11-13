# Tokens for ordering side effects

Ports of `jax.lax.create_token` and `jax.lax.after_all`. jaxR represents
a token as a host value with no array leaves (there is no token type in
its compiled programs), so tokens can be created, merged and returned
from transformed functions but order nothing.

## Usage

``` r
lax_create_token(x = NULL)

lax_after_all(...)
```

## Arguments

- x:

  Ignored (kept for compatibility, as in JAX).

- ...:

  Tokens.

## Value

A token.

## Examples

``` r
t <- lax_create_token()
lax_after_all(t, lax_create_token())
#> Token()
```
