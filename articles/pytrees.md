# Pytrees

``` r

library(jaxR)
```

A *pytree* is a nested container of arrays: the parameters of a model,
the state of an optimizer, a batch of data. Every transformation in jaxR
accepts pytrees wherever it accepts arrays. In R, the pytree containers
are lists (named or not), `NULL` (a tree with no leaves) and classes
registered with the functions of this vignette. Everything else is a
leaf.

``` r

params <- list(
  dense = list(w = jnp_ones(c(2L, 3L)), b = jnp_zeros(3L)),
  scale = as_jax(2))
```

## Pytrees in transformations

`grad` returns a tree with the structure of its argument:

``` r

loss <- function(p, x) jnp_sum((x %*% p$dense$w + p$dense$b)^2) * p$scale
g <- grad(loss)(params, jnp_ones(c(4L, 2L)))
str(lapply(g$dense, dim))
#> List of 2
#>  $ w: int [1:2] 2 3
#>  $ b: int 3
g$scale
#> Array f32[] weak_type
#> [1] 48
```

`vmap` maps over the leading axis of every leaf, `jit` flattens
arguments and results, and control flow carries pytrees as loop state:

``` r

batch <- list(a = jnp_reshape(jnp_arange(6, dtype = "float32"), c(3L, 2L)), b = as_jax(c(1, 2, 3)))
vmap(function(el) jnp_sum(el$a) * el$b)(batch)
#> Array f32[3]
#> [1]  1 10 27

lax_fori_loop(0L, 3L, function(i, s) list(n = s$n + 1L, total = s$total + i),
              list(n = 0L, total = 0L))
#> $n
#> Array i32[] weak_type
#> [1] 3
#> 
#> $total
#> Array i32[]
#> [1] 3
```

## Working with trees

``` r

tree_leaves(params)
#> [[1]]
#> Array f32[3]
#> [1] 0 0 0
#> 
#> [[2]]
#> Array f32[2,3]
#>      [,1] [,2] [,3]
#> [1,]    1    1    1
#> [2,]    1    1    1
#> 
#> [[3]]
#> Array f32[] weak_type
#> [1] 2
tree_structure(params)
#> PyTreeDef(list(dense = list(w = *, b = *), scale = *))
tree_map(function(x) x * 0, params)$dense$b
#> Array f32[3]
#> [1] 0 0 0
tree_map(function(x, y) x + y, list(1, list(2, 3)), list(10, list(20, 30)))
#> [[1]]
#> [1] 11
#> 
#> [[2]]
#> [[2]][[1]]
#> [1] 22
#> 
#> [[2]][[2]]
#> [1] 33
tree_reduce(function(acc, x) acc + jnp_size(x), params, 0L)
#> [1] 10
tree_all(tree_map(function(x) jnp_all(jnp_isfinite(x)), params))
#> [1] TRUE
```

`tree_flatten` and `tree_unflatten` are the inverse pair underneath:

``` r

fl <- tree_flatten(params)
length(fl$leaves)
#> [1] 3
rebuilt <- tree_unflatten(fl$treedef, lapply(fl$leaves, function(x) x + 1))
rebuilt$scale
#> Array f32[] weak_type
#> [1] 3
```

Named lists flatten in sorted key order, as Python dictionaries do in
JAX, so two trees with the same names in a different order have the same
leaves order:

``` r

tree_leaves(list(b = 2, a = 1))
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
```

`is_leaf` stops the traversal early:

``` r

length(tree_leaves(params, is_leaf = function(x) is.list(x) && !is.null(x$w)))
#> [1] 2
```

### Paths

``` r

kl <- tree_leaves_with_path(params)
vapply(kl, function(p) keystr(p[[1]]), "")
#> [1] "['dense']['b']" "['dense']['w']" "['scale']"
vapply(kl, function(p) keystr(p[[1]], simple = TRUE, separator = "/"), "")
#> [1] "dense/b" "dense/w" "scale"
tree_map_with_path(function(path, x) paste(keystr(path), jnp_size(x)), params)$dense$w
#> [1] "['dense']['w'] 6"
```

Path entries are `DictKey`, `SequenceKey`, `GetAttrKey` and
`FlattenedIndexKey` objects.

### Structure utilities

``` r

# A list of two records becomes a record of two lists
outer <- tree_structure(list(0, 0))
inner <- tree_structure(list(x = 0, y = 0))
tree_transpose(outer, inner, list(list(x = 1, y = 2), list(x = 3, y = 4)))
#> $x
#> $x[[1]]
#> [1] 1
#> 
#> $x[[2]]
#> [1] 3
#> 
#> 
#> $y
#> $y[[1]]
#> [1] 2
#> 
#> $y[[2]]
#> [1] 4

tree_broadcast(list(1, 2), list(list(3, 4), list(a = 5, b = 6)))
#> [[1]]
#> [[1]][[1]]
#> [1] 1
#> 
#> [[1]][[2]]
#> [1] 1
#> 
#> 
#> [[2]]
#> [[2]]$a
#> [1] 2
#> 
#> [[2]]$b
#> [1] 2

td <- tree_structure(params)
treedef_num_leaves(td)
#> [1] 3
length(treedef_children(td))
#> [1] 2
```

`ravel_pytree` flattens a tree of arrays to one vector and returns the
inverse, which is what optimizers that work on a flat vector need:

``` r

rv <- ravel_pytree(params)
dim(rv[[1]])
#> [1] 10
str(lapply(rv[[2]](rv[[1]] * 2)$dense, dim))
#> List of 2
#>  $ w: int [1:2] 2 3
#>  $ b: int 3
```

## Registering classes

An S3 object is a leaf until its class is registered.
`register_pytree_node` takes the class name, a flatten function
returning `list(children =, aux =)` and an unflatten function.

``` r

new_gaussian <- function(mean, sd, label) {
  structure(list(mean = mean, sd = sd, label = label), class = "gaussian")
}
register_pytree_node("gaussian",
  function(x) list(children = list(x$mean, x$sd), aux = x$label),
  function(aux, children) new_gaussian(children[[1]], children[[2]], aux))

d <- new_gaussian(as_jax(0), as_jax(1), "prior")
tree_leaves(d)
#> [[1]]
#> Array f32[] weak_type
#> [1] 0
#> 
#> [[2]]
#> Array f32[] weak_type
#> [1] 1
nll <- function(d, x) -jnp_sum(scipy_stats_norm_logpdf(x, d$mean, d$sd))
g <- grad(nll)(d, as_jax(c(0.5, 1.5)))
class(g)
#> [1] "gaussian"
g$mean
#> Array f32[] weak_type
#> [1] -2
```

For a list based class, `register_dataclass` is shorter: name the fields
that hold arrays and the fields that are static metadata.

``` r

register_dataclass("layer", data_fields = c("w", "b"), meta_fields = "activation")
l <- structure(list(w = jnp_ones(2L), b = as_jax(0), activation = "relu"), class = "layer")
vapply(tree_leaves_with_path(l), function(p) keystr(p[[1]]), "")
#> [1] ".w" ".b"
```

`register_pytree_with_keys` also records the path entry of each child,
and `register_static` declares a class with no leaves (configuration
objects that pass through transformations untouched).
`register_pytree_node_class` and `register_pytree_with_keys_class` look
up the methods `tree_flatten.<class>` and `tree_unflatten.<class>` by
name.

## Partial

`jax_Partial(f, ...)` binds leading arguments, like `functools.partial`.
The result is a pytree whose leaves are the bound arguments, so it can
be passed through `jit` and `vmap` as data.

``` r

scale_by <- jax_Partial(function(k, x) k * x, as_jax(3))
scale_by(as_jax(c(1, 2)))
#> Array f32[2]
#> [1] 3 6
tree_leaves(scale_by)
#> [[1]]
#> Array f32[] weak_type
#> [1] 3
jit(function(f, x) f(x))(scale_by, as_jax(c(1, 2)))
#> Array f32[2]
#> [1] 3 6
```

## NULL and static leaves

`NULL` is a tree without leaves, so optional parts of a structure can be
absent. R values that are not arrays (strings, functions) are leaves;
transformations reject them as array arguments, so keep them in `aux`
data or in a registered static class.

``` r

tree_leaves(list(1, NULL, list(NULL, 2)))
#> [[1]]
#> [1] 1
#> 
#> [[2]]
#> [1] 2
```
