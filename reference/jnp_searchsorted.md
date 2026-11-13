# Find insertion indices in a sorted array

Port of `jnp.searchsorted` (0-based indices), with the methods `"scan"`
(binary search with
[`lax_scan()`](https://adamrsophie.github.io/jaxR/reference/lax_scan.md)),
`"scan_unrolled"` (the same here), `"sort"` and `"compare_all"`.

## Usage

``` r
jnp_searchsorted(a, v, side = "left", sorter = NULL, method = "scan")
```

## Arguments

- a:

  Sorted one dimensional array.

- v:

  Query values.

- side:

  `"left"` or `"right"`.

- sorter:

  Optional 0-based indices that sort `a`.

- method:

  One of `"scan"`, `"scan_unrolled"`, `"sort"`, `"compare_all"`.

## Value

Integer array with the shape of `v`.
