# Pad an array

Port of `jnp.pad` with all NumPy modes: `"constant"`, `"edge"`,
`"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
`"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`, or a function
`(vector, iaxis_pad_width, iaxis, kwargs)` that returns the padded
vector (`iaxis` is 0-based and `kwargs` the named list of extra
arguments).

## Usage

``` r
jnp_pad(array, pad_width, mode = "constant", ...)
```

## Arguments

- array:

  Array.

- pad_width:

  Padding: a number, a `c(before, after)` pair, a list of pairs or an
  `(ndim, 2)` matrix.

- mode:

  Padding mode (string or function).

- ...:

  Mode options: `constant_values`, `end_values`, `stat_length`,
  `reflect_type`.

## Value

An array.
