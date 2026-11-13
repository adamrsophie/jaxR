# Select from several choices

Port of `jnp.select`: where several conditions hold, the first wins.

## Usage

``` r
jnp_select(condlist, choicelist, default = 0L)
```

## Arguments

- condlist:

  List of boolean arrays.

- choicelist:

  List of arrays, one per condition.

- default:

  Value where no condition holds.

## Value

An array.
