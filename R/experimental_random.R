# Stateful random number generator (port of jax/_src/random/stateful_rng.py, exported by
# jax.experimental.random). The generator holds a typed key and a counter in a Ref; each
# draw folds the counter into the key and adds one to the counter.

# Port of _canonicalize_size: NULL broadcasts the shapes of the parameters.
srng_size <- function(size, ...) {
  if (is.null(size)) return(broadcast_shapes_impl(lapply(list(...), rnd_shape)))
  as.integer(size)
}

srng_new <- function(base_key, counter) {
  structure(list(base_key = base_key, counter = counter), class = "jax_stateful_prng")
}

#' Stateful random number generator
#'
#' Ports of `jax.experimental.random.stateful_rng` and
#' `jax.experimental.random.StatefulPRNG`: a generator in the style of
#' `numpy.random.default_rng` whose state updates implicitly. It is a convenience
#' wrapper around the explicit keys of [random_key()]: the generator holds a key and a
#' counter stored in a reference (see [new_ref()]); each draw uses
#' `random_fold_in(key, counter)` and adds one to the counter.
#'
#' The generator works at the top level and inside [jit()], [vmap()] and [grad()],
#' as an argument or as a closed over value. Like any reference it cannot be returned
#' from a function transformed by [jit()], so it cannot be a carry of [lax_scan()] or
#' [lax_while_loop()].
#'
#' Methods of the generator `rng` (sizes are an integer, an integer vector or `NULL`
#' for the broadcast shape of the parameters; `dtype = NULL` is the default float or
#' integer dtype):
#' * `rng$key(shape = integer(0))`: a new key, or an array of keys of that shape.
#' * `rng$random(size = NULL, dtype = NULL)`: floats in `[0, 1)`.
#' * `rng$uniform(low = 0, high = 1, size = NULL, dtype = NULL)`.
#' * `rng$normal(loc = 0, scale = 1, size = NULL, dtype = NULL)`.
#' * `rng$integers(low, high = NULL, size = NULL, dtype = NULL)`: integers in
#'   `[low, high)`, or in `[0, low)` when `high` is `NULL`.
#' * `rng$split(num)`: one generator holding `num` independent children, to pass to
#'   [vmap()].
#' * `rng$spawn(n_children)`: a list of independent generators.
#'
#' `rng$base_key` and `rng$counter` give the key and the counter reference (the fields
#' `_base_key` and `_counter` of JAX).
#' @param seed Integer seed. May be `NULL` at the top level only; the seed is then
#'   drawn from R's random number generator (JAX takes it from the operating system),
#'   so `set.seed()` makes it reproducible.
#' @param impl Optional name of the generator implementation (for example
#'   `"threefry2x32"`).
#' @param base_key A typed scalar key from [random_key()].
#' @param counter A reference holding a scalar integer.
#' @return A generator (class `jax_stateful_prng`).
#' @examples
#' \dontrun{
#' rng <- experimental_random_stateful_rng(42)
#' rng$uniform()
#' rng$normal(size = 3L)
#' jit(function(x) x + rng$uniform())(1)
#' vmap(function(r, x) x + r$uniform())(rng$split(3L), jnp_zeros(3L))
#' }
#' @export
experimental_random_stateful_rng <- function(seed = NULL, impl = NULL) {
  if (is.null(seed)) {
    if (!identical(current_trace()$kind, "eval"))
      jax_error("When used within transformed code, jax.experimental.random.stateful_rng()",
                " requires an explicit seed to be set.", class = "jax_type_error")
    seed <- sample.int(.Machine$integer.max, 1L)
  }
  srng_new(random_key(seed, impl = impl), new_ref(0L))
}

#' @rdname experimental_random_stateful_rng
#' @export
experimental_random_StatefulPRNG <- function(base_key, counter) {
  if (!inherits(base_key, "jax_prng_key"))
    jax_error("Expected base_key to be a typed PRNG key; got ", class(base_key)[1L],
              class = "jax_value_error")
  if (!is_ref(counter))
    jax_error("Expected counter to be a scalar integer ref; got ", class(counter)[1L],
              class = "jax_value_error")
  srng_new(base_key, counter)
}

srng_key <- function(x, shape = integer(0)) {
  base_key <- .subset2(x, "base_key")
  counter <- .subset2(x, "counter")
  if (length(dim(base_key)))
    jax_error("cannot operate on split stateful generator", class = "jax_value_error")
  key <- random_fold_in(base_key, counter$get())
  counter$addupdate(1L)
  shape <- as.integer(shape)
  if (length(shape)) random_split(key, shape) else key
}

#' @export
`$.jax_stateful_prng` <- function(x, name) {
  switch(name,
    base_key = .subset2(x, "base_key"),
    counter = .subset2(x, "counter"),
    key = function(shape = integer(0)) srng_key(x, shape),
    random = function(size = NULL, dtype = NULL)
      random_uniform(srng_key(x), shape = srng_size(size), dtype = dtype),
    uniform = function(low = 0, high = 1, size = NULL, dtype = NULL)
      random_uniform(srng_key(x), srng_size(size, low, high), dtype = dtype, minval = low,
                     maxval = high),
    normal = function(loc = 0, scale = 1, size = NULL, dtype = NULL) {
      norm <- random_normal(srng_key(x), srng_size(size, loc, scale), dtype)
      jnp_astype(jnp_asarray(loc) + jnp_asarray(scale) * norm,
                 dtype %||% dtype(norm))
    },
    integers = function(low, high = NULL, size = NULL, dtype = NULL) {
      if (is.null(high)) {
        high <- low
        low <- 0L
      }
      random_randint(srng_key(x), srng_size(size, low, high), minval = low, maxval = high,
                     dtype = dtype)
    },
    split = function(num)
      srng_new(srng_key(x, num), new_ref(jnp_zeros(as.integer(num), dtype = "int"))),
    spawn = function(n_children)
      lapply(as.list(srng_key(x, n_children)), function(k) srng_new(k, new_ref(0L))),
    jax_error("StatefulPRNG has no attribute '", name, "'", class = "jax_attribute_error"))
}

#' @export
format.jax_stateful_prng <- function(x, ...) {
  paste0("StatefulPRNG(base_key=", format(.subset2(x, "base_key")), ", counter=",
         format(.subset2(x, "counter")), ")")
}

#' @export
print.jax_stateful_prng <- function(x, ...) {
  cat(format(x), "\n", sep = "")
  invisible(x)
}

# The generator is a pytree with the key and the counter reference as children, like
# the registered dataclass of JAX.
register_pytree_node("jax_stateful_prng",
  function(x) list(children = list(.subset2(x, "base_key"), .subset2(x, "counter")), aux = NULL),
  function(aux, children) srng_new(children[[1L]], children[[2L]]))
