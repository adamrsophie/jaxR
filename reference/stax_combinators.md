# Stax combinators

Ports of `stax.serial`, `stax.parallel` and `stax.shape_dependent`.
`stax_serial` composes layers in sequence (its params are the list of
the layers' params); `stax_parallel` applies one layer to each element
of a list of inputs (often between
[`stax_FanOut()`](https://adamrsophie.github.io/jaxR/reference/stax_layers.md)
and
[stax_FanInSum](https://adamrsophie.github.io/jaxR/reference/stax_layers.md));
`stax_shape_dependent` delays building a layer until the input shape is
known. A key passed to `apply_fun` as `rng = key` is split between the
layers.

## Usage

``` r
stax_serial(...)

stax_parallel(...)

stax_shape_dependent(make_layer)
```

## Arguments

- ...:

  Layers, each a pair `list(init_fun, apply_fun)`.

- make_layer:

  Function of an input shape returning a layer.

## Value

A layer `list(init_fun, apply_fun)`.

## Examples

``` r
if (FALSE) { # \dontrun{
net <- stax_serial(stax_FanOut(2L), stax_parallel(stax_Dense(3L), stax_Identity),
                   stax_FanInSum)
res <- net$init_fun(random_key(1), c(-1L, 3L))
} # }
```
