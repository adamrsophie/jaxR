# Run code with a default device

Port of the `jax.default_device` context manager. jaxR places arrays on
its single device, so this only checks that `device` is one of
[`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md)
(or `NULL`).

## Usage

``` r
default_device(device, code)
```

## Arguments

- device:

  A device description from
  [`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md),
  or `NULL`.

- code:

  Expression to evaluate.

## Value

The value of `code`.
