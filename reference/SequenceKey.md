# Key path entries

Ports of `jax.tree_util.SequenceKey`, `DictKey`, `GetAttrKey` and
`FlattenedIndexKey`. A key path (as returned by
[`tree_flatten_with_path()`](https://adamrsophie.github.io/jaxR/reference/tree_flatten_with_path.md))
is a list of these entries;
[`keystr()`](https://adamrsophie.github.io/jaxR/reference/keystr.md)
formats it. [`format()`](https://rdrr.io/r/base/format.html) gives JAX's
[`str()`](https://rdrr.io/r/utils/str.html) of an entry (`"[0]"`,
`"['a']"`, `".x"`, `"[<flat index 0>]"`).

## Usage

``` r
SequenceKey(idx)

DictKey(key)

GetAttrKey(name)

FlattenedIndexKey(key)
```

## Arguments

- idx:

  0-based position in an unnamed list.

- key:

  Name (or value) of a named list element; for `FlattenedIndexKey`, the
  0-based child position.

- name:

  Field name.

## Value

A key entry.
