# Format a key path

Port of `jax.tree_util.keystr`.

## Usage

``` r
keystr(keys, simple = FALSE, separator = "")
```

## Arguments

- keys:

  A key path (list of key entries, or values convertible to strings).

- simple:

  Use the compact representation (`"a/0"` style with `separator`).

- separator:

  String placed between entries.

## Value

A character scalar.

## Examples

``` r
params <- list(foo = list(bar = list(baz = 1, bat = list(2, 3))))
for (kl in tree_leaves_with_path(params)) print(keystr(kl[[1]]))
#> [1] "['foo']['bar']['bat'][0]"
#> [1] "['foo']['bar']['bat'][1]"
#> [1] "['foo']['bar']['baz']"
```
