# More lax elementwise operations

Ports of `jax.lax.log2` (base 2 logarithm), `jax.lax.mulhi` (the high
half of the double width product of integers) and
`jax.lax.one_minus_square` (`1 - x^2`, computed as `(1 + x) (1 - x)`
with the derivative `-2 x`).

## Usage

``` r
lax_log2(x, accuracy = NULL)

lax_mulhi(x, y)

lax_one_minus_square(x)
```

## Arguments

- x, y:

  Arrays (integers for `lax_mulhi`).

- accuracy:

  Optional accuracy request of `lax_log2()` (see
  [lax_elementwise](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md)).

## Value

An array.

## Examples

``` r
if (FALSE) { # \dontrun{
lax_log2(c(1, 2, 8))
lax_one_minus_square(0.5)
} # }
```
