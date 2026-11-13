# Save and load arrays in NumPy formats

Ports of `jnp.save`, `jnp.savez` and `jnp.load` (NumPy's `.npy` and
`.npz` formats, readable by NumPy). `bfloat16` arrays are stored as two
byte voids, as with NumPy, and restored by `jnp_load()`. `.npz` files
load as a named list of arrays.

## Usage

``` r
jnp_save(file, arr, allow_pickle = TRUE, fix_imports = TRUE)

jnp_savez(file, ...)

jnp_load(
  file,
  mmap_mode = NULL,
  allow_pickle = FALSE,
  fix_imports = TRUE,
  encoding = "ASCII",
  ...
)
```

## Arguments

- file:

  A file name or connection. `jnp_save()` adds `.npy` and `jnp_savez()`
  `.npz` to file names without the extension.

- arr:

  An array.

- allow_pickle, fix_imports, mmap_mode, encoding:

  Accepted for compatibility.

- ...:

  Arrays for `jnp_savez()` (named ones keep their names, others are
  `arr_0`, `arr_1`, ...); ignored by `jnp_load()`.

## Value

`jnp_load()` returns an array or a named list of arrays; the others
return `NULL` invisibly.
