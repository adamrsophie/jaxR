# Mutable array references

Ports of `jax.new_ref`, `jax.empty_ref`, `jax.free_ref`, `jax.freeze`,
`jax.Ref` and `jax.ds`. `new_ref(x)` makes a reference holding the array
`x`; read it with `r[...]` (NumPy indexing, `r[]` for the whole value)
or `r$get()`, and write it with `r[...] <- value`, `r$set(value, ...)`
or `r$addupdate(value, ...)`. `freeze(r)` returns the value and makes
the reference read only; `free_ref(r)` releases it. `ds()` builds a
dynamic slice index `ds(start, size, stride)` for reference indexing.

## Usage

``` r
new_ref(init_val, memory_space = NULL, kind = NULL)

empty_ref(ty, memory_space = NULL, pin = FALSE)

free_ref(ref)

freeze(ref)

Ref(...)

ds(start, size, stride = 1L)
```

## Arguments

- init_val:

  Initial value.

- memory_space, kind, pin:

  Accepted for compatibility.

- ty:

  A
  [`ShapeDtypeStruct()`](https://adamrsophie.github.io/jaxR/reference/ShapeDtypeStruct.md)
  for `empty_ref` (the value starts as zeros).

- ref:

  A reference.

- ...:

  Ignored: `Ref` names the class of references (port of `jax.Ref`) and
  cannot be called to build one.

- start, size, stride:

  Start (static or traced), static size and stride of `ds()`.

## Value

`new_ref()` and `empty_ref()` return a reference (class `jax_ref`);
`freeze()` returns an array; `free_ref()` returns `NULL` invisibly;
`ds()` returns an index.

## Details

Refs may be read and written inside
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md),
[`grad()`](https://adamrsophie.github.io/jaxR/reference/grad.md) and
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md): refs
created inside the transformed function behave as in JAX.
[`jit()`](https://adamrsophie.github.io/jaxR/reference/jit.md) also
handles refs passed as arguments or closed over (their updates become
outputs written back after each call); with other transformations,
writing an outer ref leaves it holding a traced value that is unusable
outside, as JAX forbids.
