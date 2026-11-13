# Convolution dimension numbers

Port of `jax.lax.ConvDimensionNumbers`: 0-based dimension numbers of
`(batch, feature, spatial...)` for the input and output and
`(out feature, in feature, spatial...)` for the kernel.

## Usage

``` r
lax_ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
```

## Arguments

- lhs_spec, rhs_spec, out_spec:

  Integer vectors.

## Value

An object of class `ConvDimensionNumbers`.
