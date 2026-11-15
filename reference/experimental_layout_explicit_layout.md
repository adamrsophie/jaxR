# Run a function with explicit layouts of its arguments

Port of `jax.experimental.layout.explicit_layout`: `f` runs with its
arguments in the layouts `in_layouts` (one per leaf, or one for all),
given here or as the argument `in_layouts` of each call. In JAX the
layouts become part of the types of the values (for Pallas kernels);
jaxR types have no layouts, so the layouts are checked against the
arguments and `f` runs on them unchanged.

## Usage

``` r
experimental_layout_explicit_layout(f, in_layouts = NULL)
```

## Arguments

- f:

  A function.

- in_layouts:

  Layouts of the arguments.

## Value

A function.
