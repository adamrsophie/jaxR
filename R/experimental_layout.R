# Port of jax.experimental.layout (jax/_src/layout.py and the layout parts of
# jax/_src/pjit.py): device memory layouts of arrays.
#
# A layout gives the order of the dimensions in memory, from the most major to the most
# minor. The CPU device stores arrays in row major order (the default layout); a
# constraint inside a compiled function lets XLA choose another order for an
# intermediate value, which changes no value.

#' Device memory layouts
#'
#' Ports of `jax.experimental.layout.Layout`, `Layout.AUTO`, `Layout.for_array`,
#' `Layout.update` and `Format`. A layout lists the dimensions of an array from the most
#' major to the most minor (0 based): `c(0, 1)` is the row major order of a matrix and
#' `c(1, 0)` the column major one. `experimental_layout_Layout_AUTO` lets the compiler
#' choose. A `Format` pairs a layout with a sharding; jaxR runs on one device, so the
#' sharding is `NULL` or a device name from [devices()].
#' @param major_to_minor Integer vector, a permutation of `0:(ndim - 1)`.
#' @param tiling `NULL` or a list of integer vectors (tiles). The CPU device has no
#'   tiled layouts, so only `NULL` and the empty list can be used in a computation.
#' @param sub_byte_element_size_in_bits Integer.
#' @param x An array (for `for_array`) or a layout (for `update`).
#' @param ... Fields to replace (`major_to_minor`, `tiling`,
#'   `sub_byte_element_size_in_bits`).
#' @param layout `NULL`, `experimental_layout_Layout_AUTO` or a layout.
#' @param sharding `NULL` or a device name.
#' @return A `jax_Layout` or `jax_Format` object.
#' @examples
#' \dontrun{
#' l <- experimental_layout_Layout(c(1L, 0L))
#' l
#' experimental_layout_Layout_for_array(jnp_ones(c(2L, 3L)))
#' experimental_layout_Format(l, devices()[1])
#' }
#' @export
experimental_layout_Layout <- function(major_to_minor, tiling = NULL,
                                       sub_byte_element_size_in_bits = 0L) {
  structure(list(major_to_minor = as.integer(major_to_minor),
                 tiling = if (!is.null(tiling)) lapply(tiling, as.integer),
                 sub_byte_element_size_in_bits = as.integer(sub_byte_element_size_in_bits)),
            class = "jax_Layout")
}

#' @rdname experimental_layout_Layout
#' @export
experimental_layout_Layout_AUTO <- structure(list(), class = "jax_AutoLayout")

#' @rdname experimental_layout_Layout
#' @export
experimental_layout_Layout_for_array <- function(x) {
  experimental_layout_Layout(seq_len(length(aval_of(as_jax(x))$shape)) - 1L, tiling = list(),
                             sub_byte_element_size_in_bits = 0L)
}

#' @rdname experimental_layout_Layout
#' @export
experimental_layout_Layout_update <- function(x, ...) {
  kw <- list(...)
  bad <- setdiff(names(kw), c("major_to_minor", "tiling", "sub_byte_element_size_in_bits"))
  if (length(bad) || (length(kw) && is.null(names(kw))))
    jax_error("Layout.update got unexpected keyword arguments: ", paste(bad, collapse = ", "),
              class = "jax_type_error")
  f <- unclass(x)
  f[names(kw)] <- kw
  experimental_layout_Layout(f$major_to_minor, f$tiling, f$sub_byte_element_size_in_bits)
}

#' @rdname experimental_layout_Layout
#' @export
experimental_layout_Format <- function(layout = NULL, sharding = NULL) {
  if (inherits(layout, "jax_Layout") && is.null(sharding))
    jax_error("Sharding has to be concrete when layout is of type <class 'Layout'>. Please ",
              "pass a device to the sharding argument. Got sharding None",
              class = "jax_value_error")
  if (!is.null(layout) && !inherits(layout, c("jax_Layout", "jax_AutoLayout")))
    jax_error("Invalid value received for the layout argument. Expected values are `None`, ",
              "`Layout.AUTO` or an instance of `Layout`. Got ", class(layout)[1L],
              class = "jax_type_error")
  if (!is.null(sharding) && !(is.character(sharding) && length(sharding) == 1L &&
                               sharding %in% devices()))
    jax_error("Invalid value received for the sharding argument. Expected values are `None` ",
              "or a device name from devices(). Got ", format(sharding)[1L],
              class = "jax_type_error")
  structure(list(layout = layout, sharding = sharding), class = "jax_Format")
}

layout_tup <- function(v) if (length(v) == 1L) paste0("(", v, ",)") else paste0("(", paste(v, collapse = ", "), ")")

layout_str <- function(l) {
  if (inherits(l, "jax_AutoLayout")) return("AutoLayout")
  if (is.null(l)) return("None")
  tup <- layout_tup
  til <- if (is.null(l$tiling)) "None"
         else if (!length(l$tiling)) "()"
         else tup(vapply(l$tiling, tup, ""))
  paste0("Layout(major_to_minor=", tup(l$major_to_minor), ", tiling=", til,
         ", sub_byte_element_size_in_bits=", l$sub_byte_element_size_in_bits, ")")
}

#' @export
print.jax_Layout <- function(x, ...) {
  cat(layout_str(x), "\n", sep = "")
  invisible(x)
}

#' @export
print.jax_AutoLayout <- function(x, ...) {
  cat("AutoLayout\n")
  invisible(x)
}

#' @export
print.jax_Format <- function(x, ...) {
  cat("Format(layout=", layout_str(x$layout), ", sharding=", x$sharding %||% "None", ")\n", sep = "")
  invisible(x)
}

#' @export
Ops.jax_Layout <- function(e1, e2) {
  eq <- inherits(e2, "jax_Layout") && inherits(e1, "jax_Layout") &&
    identical(unclass(e1), unclass(e2))
  switch(.Generic, "==" = eq, "!=" = !eq,
         jax_error("unsupported operator ", .Generic, " for Layout", class = "jax_type_error"))
}

# Port of check_aval_layout_compatibility.
layout_check_aval <- function(l, aval, what) {
  if (length(l$major_to_minor) != length(aval$shape))
    jax_error("Length of major_to_minor and the rank of the value should match. Got ",
              "major_to_minor=", layout_tup(l$major_to_minor), " and shape=",
              layout_tup(aval$shape), " for ", what, class = "jax_value_error")
  if (!setequal(l$major_to_minor, seq_along(aval$shape) - 1L))
    jax_error("major_to_minor must be a permutation of the dimensions, got ",
              layout_tup(l$major_to_minor), class = "jax_value_error")
  if (length(l$tiling))
    jax_error("tiled layouts are not supported on the CPU device", class = "jax_not_implemented_error")
}

# ---- with_layout_constraint ----------------------------------------------------------

layout_constraint_p <- new_primitive("layout_constraint")
layout_constraint_p$abstract_eval <- function(avals, params) avals[[1L]]
layout_constraint_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  a <- avals_in[[1L]]
  rank <- length(a$shape)
  m2m <- params$layout$major_to_minor
  idx <- function(v) {
    if (!length(v)) return("dense<> : tensor<0xindex>")
    paste0("dense<[", paste(v, collapse = ", "), "]> : tensor<", length(v), "xindex>")
  }
  t <- tensor_type(a)
  v <- ssa_name(ctx)
  # Port of mlir.wrap_with_layout_op: a LayoutConstraint custom call whose result has
  # the layout (minor to major); XLA ignores the operand layout.
  emit_line(ctx, paste0(
    v, " = \"stablehlo.custom_call\"(", ins[[1L]], ") <{api_version = 1 : i32, ",
    "call_target_name = \"LayoutConstraint\", operand_layouts = [", idx(seq_len(rank) - 1L),
    "], result_layouts = [", idx(rev(m2m)), "]}> : (", t, ") -> ", t))
  v
}
deflinear(layout_constraint_p, function(ct, args, params) list(bind(layout_constraint_p, list(ct), params)))
# Port of get_layout_for_vmap: the batch dimension becomes the most major.
layout_constraint_p$batch <- function(vals, dims, params, axis) {
  d <- dims[1L]
  if (is.na(d)) return(list(bind(layout_constraint_p, vals, params), d))
  m2m <- params$layout$major_to_minor + 1L
  m2m <- append(m2m, d, after = 0L)
  l <- experimental_layout_Layout_update(params$layout, major_to_minor = m2m)
  list(bind(layout_constraint_p, vals, list(layout = l)), d)
}

#' Constrain the memory layout of values inside a computation
#'
#' Port of `jax.experimental.layout.with_layout_constraint`: the leaves of `x` get the
#' layouts `layouts` (one layout, or a tree of layouts matching `x`) inside a compiled
#' function, which lets XLA store an intermediate value in another dimension order.
#' Values are unchanged; outside [jit()] the arrays keep the row major layout of the CPU
#' device.
#' @param x A pytree of arrays.
#' @param layouts A layout from [experimental_layout_Layout()], or a pytree of them.
#' @return `x`, with the constraints.
#' @examples
#' \dontrun{
#' f <- jit(function(x) experimental_layout_with_layout_constraint(x * 2,
#'   experimental_layout_Layout(c(1L, 0L))))
#' f(jnp_ones(c(2L, 3L)))
#' }
#' @export
experimental_layout_with_layout_constraint <- function(x, layouts) {
  fl <- tree_flatten(x)
  leaves <- lapply(fl$leaves, as_jax)
  ls <- if (inherits(layouts, "jax_Layout")) rep(list(layouts), length(leaves))
        else flatten_up_to(fl$treedef, layouts)
  if (!all(vapply(ls, inherits, NA, "jax_Layout")))
    jax_error("layouts passed to `with_layout_constraint` must be of type `Layout`. Got ",
              paste(vapply(ls, function(l) class(l)[1L], ""), collapse = ", "),
              class = "jax_value_error")
  for (i in seq_along(leaves))
    layout_check_aval(ls[[i]], aval_of(leaves[[i]]), "with_layout_constraint arguments")
  tree_unflatten(fl$treedef, Map(function(v, l) bind(layout_constraint_p, list(v), list(layout = l)),
                                 leaves, ls))
}

#' Run a function with explicit layouts of its arguments
#'
#' Port of `jax.experimental.layout.explicit_layout`: `f` runs with its arguments in the
#' layouts `in_layouts` (one per leaf, or one for all), given here or as the argument
#' `in_layouts` of each call. In JAX the layouts become part of the types of the values
#' (for Pallas kernels); jaxR types have no layouts, so the layouts are checked against
#' the arguments and `f` runs on them unchanged.
#' @param f A function.
#' @param in_layouts Layouts of the arguments.
#' @return A function.
#' @export
experimental_layout_explicit_layout <- function(f, in_layouts = NULL) {
  force(f)
  function(...) {
    args <- list(...)
    lays <- in_layouts
    if (is.null(lays)) {
      if (!"in_layouts" %in% names(args))
        jax_error("Missing required keyword argument: 'in_layouts'", class = "jax_type_error")
      lays <- args$in_layouts
      args$in_layouts <- NULL
    }
    fl <- tree_flatten(args)
    ls <- if (inherits(lays, c("jax_Layout", "jax_AutoLayout")) || is.null(lays))
            rep(list(lays), length(fl$leaves))
          else flatten_up_to(fl$treedef, if (length(args) == 1L && !is.list(lays[[1L]])) list(lays) else lays,
                             null_ok = TRUE)
    kinds <- unique(vapply(ls, function(l) class(l %||% list())[1L], ""))
    if (length(kinds) > 1L)
      jax_error("All args passed to `explicit_layout` must have the same type of layout.",
                class = "jax_type_error")
    for (i in seq_along(ls))
      if (inherits(ls[[i]], "jax_Layout"))
        layout_check_aval(ls[[i]], aval_of(as_jax(fl$leaves[[i]])), "explicit_layout arguments")
    do.call(f, args)
  }
}
