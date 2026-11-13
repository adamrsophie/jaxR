# Register an S3 list class with fixed fields as a pytree node

Port of `jax.tree_util.register_dataclass` for S3 classes built on named
lists: `data_fields` become children (with
[`GetAttrKey()`](https://adamrsophie.github.io/jaxR/reference/SequenceKey.md)
paths), `meta_fields` are static data kept in the treedef. R has no
dataclass declarations, so both field lists are required, as JAX
requires them for classes that are not dataclasses. Unflattening
rebuilds the list with the data and meta fields (in their original
order) and the original class attribute.

## Usage

``` r
register_dataclass(
  nodetype,
  data_fields = NULL,
  meta_fields = NULL,
  drop_fields = character(0)
)
```

## Arguments

- nodetype:

  Class name.

- data_fields, meta_fields:

  Character vectors of field names.

- drop_fields:

  Accepted for compatibility (fields outside `data_fields` and
  `meta_fields` are always dropped).

## Value

`nodetype`, invisibly.
