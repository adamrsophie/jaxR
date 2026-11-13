# Precision and algorithm settings of lax functions

R counterparts of the enum like classes of `jax.lax`:

- `lax_Precision`: the names `"DEFAULT"`, `"HIGH"` and `"HIGHEST"` (port
  of `jax.lax.Precision`), accepted by the `precision` argument of
  [`lax_dot_general()`](https://adamrsophie.github.io/jaxR/reference/lax_dot_general.md),
  [`lax_dot()`](https://adamrsophie.github.io/jaxR/reference/lax_dot.md)
  and
  [`lax_conv_general_dilated()`](https://adamrsophie.github.io/jaxR/reference/lax_conv_general_dilated.md).

- `lax_RoundingMethod`: `AWAY_FROM_ZERO = 0L`, `TO_NEAREST_EVEN = 1L`,
  the values of the `rounding_method` argument of
  [`lax_round()`](https://adamrsophie.github.io/jaxR/reference/lax_elementwise.md).

- `lax_AccuracyMode`: `HIGHEST = 1L`, `DEFAULT = 2L` (port of
  `jax.lax.AccuracyMode`).

- `lax_DotAlgorithmPreset`: the named presets of
  `jax.lax.DotAlgorithmPreset`.

## Usage

``` r
lax_Precision

lax_RoundingMethod

lax_AccuracyMode

lax_DotAlgorithmPreset
```

## Format

Named lists.

## Examples

``` r
lax_Precision$HIGHEST
#> [1] "HIGHEST"
lax_RoundingMethod$TO_NEAREST_EVEN
#> [1] 1
```
