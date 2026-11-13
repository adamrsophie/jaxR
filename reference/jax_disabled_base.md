# Base functions that are disabled for jax values

[`c()`](https://rdrr.io/r/base/c.html),
[`rev()`](https://rdrr.io/r/base/rev.html),
[`rep()`](https://rdrr.io/r/base/rep.html),
[`aperm()`](https://rdrr.io/r/base/aperm.html), `[[`, `dim<-` and
`length<-` raise errors for jax values because they would silently drop
dtype information or use R's column-major or 1-based conventions. Use
[`jnp_concatenate()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md),
[`jnp_flip()`](https://adamrsophie.github.io/jaxR/reference/jnp_flip.md),
[`jnp_tile()`](https://adamrsophie.github.io/jaxR/reference/jnp_tile.md),
[`jnp_transpose()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md),
`x[i]` (0-based) and
[`jnp_reshape()`](https://adamrsophie.github.io/jaxR/reference/jnp_shape.md),
or convert explicitly with
[`as.array()`](https://rdrr.io/r/base/array.html).

## Usage

``` r
# S3 method for class 'jax_value'
c(x, ...)

# S3 method for class 'jax_value'
rev(x, ...)

# S3 method for class 'jax_value'
rep(x, ...)

# S3 method for class 'jax_value'
aperm(a, perm, ...)

# S3 method for class 'jax_value'
x[[...]]

# S3 method for class 'jax_value'
dim(x) <- value

# S3 method for class 'jax_value'
length(x) <- value
```

## Arguments

- x:

  A jax value.

- ...:

  Ignored.

- a, perm:

  Ignored.

- value:

  Ignored.
