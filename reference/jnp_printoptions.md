# Print options

Ports of `jnp.get_printoptions`, `jnp.set_printoptions` and
`jnp.printoptions` (NumPy's print options, which JAX uses to print
arrays). The options are stored per session with NumPy's defaults;
`jnp_printoptions` sets them while `code` runs.

## Usage

``` r
jnp_get_printoptions()

jnp_set_printoptions(
  precision = NULL,
  threshold = NULL,
  edgeitems = NULL,
  linewidth = NULL,
  suppress = NULL,
  nanstr = NULL,
  infstr = NULL,
  formatter = NULL,
  sign = NULL,
  floatmode = NULL,
  legacy = NULL,
  override_repr = NULL
)

jnp_printoptions(code, ...)
```

## Arguments

- precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr,
  formatter, sign, floatmode, legacy, override_repr:

  NumPy print options; `NULL` leaves an option unchanged.

- code:

  Code to evaluate with the options set.

- ...:

  Options for `jnp_printoptions`, as for `jnp_set_printoptions`.

## Value

`jnp_get_printoptions` returns the list of options;
`jnp_set_printoptions` returns the previous options invisibly;
`jnp_printoptions` returns the value of `code`.
