# Reductions that ignore NaN

Ports of `jnp.nansum`, `jnp.nanprod`, `jnp.nanmax`, `jnp.nanmin`,
`jnp.nanmean`, `jnp.nanvar`, `jnp.nanstd`, `jnp.nanargmax` and
`jnp.nanargmin`. NaN entries are skipped; integer and boolean inputs
reduce like the plain reductions. `nanmax` and `nanmin` (without
`initial`) give NaN where every entry is NaN, `nanargmax` and
`nanargmin` give -1 there.

## Usage

``` r
jnp_nanmin(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_nanmax(
  a,
  axis = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_nansum(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_nanprod(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_nanmean(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  where = NULL
)

jnp_nanvar(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  ddof = 0L,
  keepdims = FALSE,
  where = NULL,
  mean = NULL
)

jnp_nanstd(
  a,
  axis = NULL,
  dtype = NULL,
  out = NULL,
  ddof = 0L,
  keepdims = FALSE,
  where = NULL,
  mean = NULL
)

jnp_nanargmax(a, axis = NULL, out = NULL, keepdims = NULL)

jnp_nanargmin(a, axis = NULL, out = NULL, keepdims = NULL)
```

## Arguments

- a:

  Array.

- axis:

  0-based axis or axes to reduce; `NULL` reduces all. `nanargmax` and
  `nanargmin` take a single axis.

- out:

  Not supported (must be `NULL`).

- keepdims:

  Keep reduced dimensions as size one.

- initial:

  Optional scalar starting value of the reduction.

- where:

  Optional boolean mask of the elements to include.

- dtype:

  Optional accumulation and result dtype.

- ddof:

  Delta degrees of freedom.

- mean:

  Optional precomputed mean (broadcastable against `a` with kept dims).

## Value

An array.
