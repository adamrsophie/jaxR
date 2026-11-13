# NumPy style universal functions

Port of `jax.numpy.ufunc`. `jnp_ufunc()` wraps an elementwise function
`func` of `nin` arrays returning `nout` arrays (a list when `nout > 1`)
into a callable ufunc object; calling it broadcasts the inputs and maps
`func` over every element (with
[`vmap()`](https://adamrsophie.github.io/jaxR/reference/vmap.md)) unless
a vectorized `call` is supplied. `jnp_frompyfunc()` is `jnp.frompyfunc`.
The ufunc methods of JAX are the functions `jnp_ufunc_reduce()`
(`ufunc.reduce`), `jnp_ufunc_accumulate()`, `jnp_ufunc_reduceat()`,
`jnp_ufunc_outer()` and `jnp_ufunc_at()`; they are also reachable with
`$`, as in `jnp_ufuncs$add$reduce(x)`, and `u$nin`, `u$nout`, `u$nargs`,
`u$identity` and `u$name` give the properties.

## Usage

``` r
jnp_ufunc(
  func,
  nin,
  nout,
  name = NULL,
  nargs = NULL,
  identity = NULL,
  call = NULL,
  reduce = NULL,
  accumulate = NULL,
  at = NULL,
  reduceat = NULL
)

jnp_frompyfunc(func, nin, nout, identity = NULL)

jnp_ufunc_reduce(
  ufunc,
  a,
  axis = 0L,
  dtype = NULL,
  out = NULL,
  keepdims = FALSE,
  initial = NULL,
  where = NULL
)

jnp_ufunc_accumulate(ufunc, a, axis = 0L, dtype = NULL, out = NULL)

jnp_ufunc_reduceat(ufunc, a, indices, axis = 0L, dtype = NULL, out = NULL)

jnp_ufunc_outer(ufunc, A, B)

jnp_ufunc_at(ufunc, a, indices, b = NULL, inplace = TRUE)

# S3 method for class 'jnp_ufunc'
x$name

# S3 method for class 'jnp_ufunc'
print(x, ...)
```

## Arguments

- func:

  Elementwise function of `nin` arguments.

- nin, nout:

  Number of inputs and outputs.

- name:

  Name of the ufunc (by default the name of `func` or `"<lambda>"`); for
  `$`, the property or method name.

- nargs:

  Number of arguments (default `nin`).

- identity:

  Identity element of a binary ufunc, or `NULL`.

- call, reduce, accumulate, at, reduceat:

  Optional specialized implementations.

- ufunc:

  A ufunc object.

- a, A, B:

  Arrays.

- axis:

  Axis (0-based); `NULL` reduces over all axes in `jnp_ufunc_reduce()`.

- dtype:

  Optional result dtype.

- out:

  Not supported (must be `NULL`).

- keepdims:

  Keep reduced dimensions with size one.

- initial:

  Initial value of the reduction.

- where:

  Boolean mask of the elements to reduce.

- indices:

  Indices (see Details).

- b:

  Second operand of `at` for binary ufuncs.

- inplace:

  Must be `FALSE` (JAX arrays are immutable).

- x:

  A ufunc object.

- ...:

  Ignored.

## Value

`jnp_ufunc()` and `jnp_frompyfunc()` return a ufunc object; the methods
return arrays.

## Details

Axes are 0-based. `indices` of `jnp_ufunc_at()` is one index (see
[`at_get()`](https://adamrsophie.github.io/jaxR/reference/at.md)) or a
plain list of indices, one per dimension (a Python tuple). As in JAX,
`at` cannot operate in place: pass `inplace = FALSE` and use the result.

## Examples

``` r
if (FALSE) { # \dontrun{
jnp_ufuncs$add$reduce(jnp_arange(5))
u <- jnp_frompyfunc(function(x, y) x * y + 1, nin = 2, nout = 1, identity = 0)
jnp_ufunc_outer(u, c(1, 2), c(3, 4, 5))
} # }
```
