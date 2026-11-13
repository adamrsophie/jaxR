# Concrete arrays (port of jax.Array): device buffers with a lazily fetched host copy.

# `host` is R data with dim = shape (no dim for rank 0) holding the exact values.
new_array <- function(aval, buf = NULL, host = NULL) {
  e <- new.env(parent = emptyenv())
  e$aval <- aval
  e$buf <- buf
  e$host <- host
  class(e) <- c("jax_array", "jax_value")
  if (.jx$track_live) live_register(e)
  e
}

live_register <- function(e) {
  n <- .jx$live_n + 1L
  .jx$live_n <- n
  assign(as.character(n), .Call(jaxr_weakref_new, e), envir = .jx$live)
  if (n %% 4096L == 0L) live_prune()
}

live_prune <- function() {
  keys <- ls(.jx$live, sorted = FALSE)
  for (k in keys) if (is.null(.Call(jaxr_weakref_get, get(k, envir = .jx$live)))) rm(list = k, envir = .jx$live)
  invisible(keys)
}

#' Live arrays
#'
#' Port of `jax.live_arrays`: the jax arrays that are still reachable from R (arrays
#' freed by R's garbage collector are not listed; call [gc()] first to collect garbage).
#' @param platform Ignored (one CPU device).
#' @return A list of jax arrays.
#' @export
live_arrays <- function(platform = NULL) {
  live_prune()
  keys <- ls(.jx$live, sorted = FALSE)
  keys <- keys[order(as.integer(keys))]
  out <- lapply(keys, function(k) .Call(jaxr_weakref_get, get(k, envir = .jx$live)))
  Filter(function(a) !is.null(a) && !isTRUE(a$deleted), out)
}

array_buf <- function(x) {
  if (is.null(x$buf)) {
    dt <- x$aval$dtype
    if (is_ext_dtype(dt)) {
      wide <- new_array(new_aval(x$aval$shape, if (is_float_dtype(dt)) "float64" else "int32"), host = x$host)
      x$buf <- array_buf(lax_convert_element_type(wide, dt))
    } else if (dt == "float0") {
      # float0 has no device type: it travels as booleans, as it is lowered.
      s <- x$aval$shape
      x$buf <- buffer_from_r(with_shape(logical(prod_int(s)), s), "bool", s)
    } else {
      x$buf <- buffer_from_r(x$host, dt, x$aval$shape)
    }
  }
  x$buf
}

array_host <- function(x) {
  if (is.null(x$host)) {
    a <- x$aval
    if (a$dtype == "float0") {
      x$host <- with_shape(logical(prod_int(a$shape)), a$shape)
    } else if (is_ext_dtype(a$dtype)) {
      # The transfer code has no narrow types: widen on the device first (exact).
      wide <- lax_convert_element_type(x, if (is_float_dtype(a$dtype)) "float64" else "int32")
      x$host <- array_host(wide)
    } else {
      x$host <- buffer_to_r(x$buf, a$dtype, a$shape)
    }
  }
  x$host
}

with_shape <- function(data, shape) {
  if (length(shape) == 0L) {
    attr(data, "dim") <- NULL
  } else {
    dim(data) <- shape
  }
  data
}

is_jax_array <- function(x) inherits(x, "jax_array")
is_tracer <- function(x) inherits(x, "jax_tracer")
is_jax_value <- function(x) inherits(x, "jax_value")

r_shape <- function(x) {
  d <- dim(x)
  if (!is.null(d)) return(as.integer(d))
  if (length(x) == 1L) integer(0) else length(x)
}

# Host data is exact for these (R data dtype, target dtype) pairs.
host_exact <- function(src, dt) {
  (src == dt) || (src == "int32" && dt %in% c("int64", "float64")) ||
    (src == "bool" && dt == "bool")
}

#' Convert R data to a jax array
#'
#' Port of `jnp.asarray` for R inputs. R doubles become `float32` arrays unless x64
#' is enabled, R integers `int32`, logicals `bool`, complex values `complex64`
#' (`complex128` with x64). A length one vector without `dim` is a rank 0 array; R
#' numeric scalars are weakly typed, as Python scalars are in JAX.
#' @param x R data, a jax array, or a tracer.
#' @param dtype Optional dtype.
#' @return A jax array (or tracer).
#' @export
as_jax <- function(x, dtype = NULL) {
  if (is_jax_value(x)) {
    if (is.null(dtype)) return(x)
    dt <- dtype(dtype)
    if (aval_of(x)$dtype == dt && !aval_of(x)$weak) return(x)
    return(lax_convert_element_type(x, dt))
  }
  if (is.factor(x) || is.list(x) || is.character(x) || is.function(x) || is.null(x) ||
      is.environment(x))
    jax_error("cannot convert object of class '", class(x)[1L], "' to a jax array",
              class = "jax_type_error")
  if (inherits(x, "sym")) return(sym_to_jax_constant(x, dtype))
  if (!is.complex(x) && anyNA(x) && any(is.na(x) & !is.nan(x)))
    jax_error("NA values cannot be converted to jax arrays (NaN can)")
  shape <- r_shape(x)
  src <- if (is.logical(x)) "bool" else if (is.integer(x)) "int32"
         else if (is.double(x)) "float64" else if (is.complex(x)) "complex128"
         else jax_error("cannot convert R data of type '", typeof(x), "'")
  weak <- is.null(dtype) && length(shape) == 0L && src != "bool"
  # A weak R integer scalar behaves like a Python int: default int width (int64 with x64).
  dt <- if (!is.null(dtype)) dtype(dtype) else if (weak && src == "int32") default_int()
        else canonicalize_dtype(src)
  data <- as.vector(x)
  if (is.integer(data)) storage.mode(data) <- "double"
  # R numbers given an integer dtype convert like Python numbers in NumPy: truncated
  # toward zero, and a value outside the range of the dtype is an error (the device
  # conversion would saturate it silently). The 2 and 4 bit integers of ml_dtypes wrap.
  if (!is.null(dtype) && dtype_kind(dt) %in% c("i", "u") && src %in% c("int32", "float64")) {
    if (anyNA(data)) jax_error("cannot convert float NaN to integer", class = "jax_value_error")
    if (any(is.infinite(data)))
      jax_error("cannot convert float infinity to integer",
                class = c("jax_overflow_error", "jax_value_error"))
    data <- trunc(data)
    r <- iinfo(dt)
    # The upper bound is exclusive: the largest 64 bit integer is not an R number.
    bad <- if (r$bits >= 8L) which(data < r$min | data >= 2^(r$bits - (r$min < 0)))
    if (r$bits < 8L) data <- (data - r$min) %% 2^r$bits + r$min
    if (length(bad))
      jax_error("Python integer ", format(data[bad[1L]], scientific = FALSE, trim = TRUE),
                " out of bounds for ", dt, class = c("jax_overflow_error", "jax_value_error"))
  }
  data <- with_shape(data, shape)
  if (host_exact(src, dt) || (src == "float64" && dt == "float64")) {
    return(new_array(new_aval(shape, dt, weak), host = data))
  }
  same_kind <- !is_ext_dtype(dt) && (dtype_kind(dt) == "f" && src %in% c("float64", "int32")) ||
    (dtype_kind(dt) == "c" && src %in% c("complex128", "float64", "int32")) ||
    (dt == "int32" && src == "int32")
  if (same_kind) {
    if (dtype_kind(dt) == "c" && !is.complex(data)) data <- with_shape(as.complex(data), shape)
    return(new_array(new_aval(shape, dt, weak), buf = buffer_from_r(data, dt, shape)))
  }
  # Cross-kind conversions use XLA semantics (saturation, truncation, x != 0).
  base_dt <- if (src == "complex128") "complex128" else if (src == "bool") "bool"
             else if (src == "int32" && is_int_dtype(dt)) "int32" else "float64"
  base <- new_array(new_aval(shape, base_dt, FALSE), host = data)
  out <- lax_convert_element_type(base, dt)
  if (weak) out <- new_array(new_aval(shape, dt, TRUE), buf = array_buf(out))
  out
}

#' Move data to the device
#'
#' Port of `jax.device_put` (single device).
#' @param x A pytree of R data or arrays.
#' @param device Ignored (one device).
#' @export
device_put <- function(x, device = NULL) {
  force(x)
  old <- .jx$explicit_transfer
  .jx$explicit_transfer <- TRUE
  on.exit(.jx$explicit_transfer <- old, add = TRUE)
  tree_map(function(l) {
    a <- as_jax(l)
    if (is_jax_array(a)) array_buf(a)
    a
  }, x)
}

#' Wait for an array computation to finish
#' @param x A pytree of arrays.
#' @export
block_until_ready <- function(x) {
  for (l in tree_leaves(x)) if (is_jax_array(l)) array_host(l)
  invisible(x)
}

#' Copy an array to the host
#'
#' Port of `jax.device_get`: returns plain R data (arrays keep their shape).
#' @param x A pytree of arrays.
#' @export
device_get <- function(x) {
  force(x)
  old <- .jx$explicit_transfer
  .jx$explicit_transfer <- TRUE
  on.exit(.jx$explicit_transfer <- old, add = TRUE)
  tree_map(function(l) if (is_jax_array(l)) array_host(l) else l, x)
}

# ---- avals ------------------------------------------------------------------------

new_aval <- function(shape, dtype, weak = FALSE) {
  structure(list(shape = as.integer(shape), dtype = dtype, weak = isTRUE(weak)),
            class = "jax_aval")
}

aval_of <- function(x) {
  if (is_tracer(x) || is_jax_array(x)) return(x$aval)
  if (inherits(x, "jax_aval")) return(x)
  aval_of(as_jax(x))
}

aval_str <- function(a) {
  short <- c(bool = "bool", int8 = "i8", int16 = "i16", int32 = "i32", int64 = "i64",
             uint8 = "u8", uint16 = "u16", uint32 = "u32", uint64 = "u64", float16 = "f16",
             bfloat16 = "bf16", float32 = "f32", float64 = "f64", complex64 = "c64",
             complex128 = "c128", float0 = "float0")
  nm <- if (a$dtype %in% names(short)) short[[a$dtype]] else
    sub("^complex", "c", sub("^int", "i", sub("^uint", "u", sub("^float", "f", a$dtype))))
  paste0(nm, "[", paste(a$shape, collapse = ","), "]")
}

#' @export
print.jax_aval <- function(x, ...) {
  cat("ShapedArray(", aval_str(x), if (x$weak) ", weak_type=True", ")\n", sep = "")
  invisible(x)
}

#' @export
format.jax_aval <- function(x, ...) aval_str(x)

aval_eq <- function(a, b) identical(a$shape, b$shape) && identical(a$dtype, b$dtype)

#' Shape and dtype of a value without computing it
#'
#' Port of `jax.ShapeDtypeStruct`.
#' @param shape Integer vector.
#' @param dtype Dtype.
#' @param weak_type Logical.
#' @export
ShapeDtypeStruct <- function(shape, dtype, weak_type = FALSE) {
  new_aval(shape, dtype(dtype), weak_type)
}

# ---- C order helpers --------------------------------------------------------------

c_ravel <- function(x) {
  d <- dim(x)
  if (length(d) <= 1L) return(as.vector(x))
  as.vector(base::aperm(x, length(d):1L))
}

c_unravel <- function(v, shape) {
  r <- length(shape)
  if (r == 0L) return(as.vector(v))
  if (r == 1L) return(array(v, shape))
  base::aperm(array(v, rev(shape)), r:1L)
}

broadcast_shapes_impl <- function(shapes) {
  r <- max(0L, vapply(shapes, length, 1L))
  out <- rep(1L, r)
  for (s in shapes) {
    s <- c(rep(1L, r - length(s)), as.integer(s))
    for (i in seq_len(r)) {
      if (s[i] == out[i] || s[i] == 1L) next
      if (out[i] == 1L) out[i] <- s[i]
      else jax_error("Incompatible shapes for broadcasting: ",
                     paste(vapply(shapes, function(z) paste0("(", paste(z, collapse = ", "), ")"), ""),
                           collapse = ", "), class = "jax_value_error")
    }
  }
  out
}

#' Broadcast shapes
#' @param ... Integer shape vectors.
#' @export
broadcast_shapes <- function(...) broadcast_shapes_impl(list(...))

# ---- conversions to R -------------------------------------------------------------

# `kind` selects the condition class (see ?jax_errors): "array", "bool" and "integer"
# are the conversions of a traced value to R data.
concrete_or_error <- function(x, what = "this operation", kind = "concretization") {
  if (is_tracer(x)) {
    v <- tracer_concrete_value(x)
    if (!is.null(v)) return(v)
    cls <- switch(kind, array = "jax_tracer_array_conversion_error",
                  bool = "jax_tracer_bool_conversion_error",
                  integer = "jax_tracer_integer_conversion_error", NULL)
    jax_error("Abstract tracer value encountered where concrete value is expected: ",
              format(x), ". The problem arose with ", what, ". A traced value cannot be ",
              "converted to R data or used in R control flow (if, while) inside jit, ",
              "vmap or other staging transformations; use lax_cond / lax_while_loop.",
              class = c(cls, "jax_concretization_error", "jax_type_error"))
  }
  as_jax(x)
}

#' @export
as.array.jax_value <- function(x, ...) array_host(concrete_or_error(x, "as.array()", "array"))

#' @export
as.vector.jax_value <- function(x, mode = "any") {
  as.vector(array_host(concrete_or_error(x, "as.vector()", "array")), mode)
}

#' @export
as.double.jax_value <- function(x, ...) as.double(as.vector(x))

#' @export
as.numeric.jax_value <- as.double.jax_value

#' @export
as.integer.jax_value <- function(x, ...) {
  as.integer(as.vector(array_host(concrete_or_error(x, "as.integer()", "integer"))))
}

#' @export
as.logical.jax_value <- function(x, ...) {
  as.logical(as.vector(array_host(concrete_or_error(x, "as.logical()", "bool"))))
}

#' @export
as.complex.jax_value <- function(x, ...) as.complex(as.vector(x))

#' @export
as.matrix.jax_value <- function(x, ...) as.matrix(as.array(x))

#' @export
format.jax_array <- function(x, ...) paste0("Array(", aval_str(x$aval), ")")

#' @export
print.jax_array <- function(x, ...) {
  a <- x$aval
  cat("Array ", aval_str(a), if (a$weak) " weak_type", "\n", sep = "")
  d <- array_host(x)
  digits <- switch(a$dtype, float16 = , bfloat16 = 4L, float32 = , complex64 = 7L, NULL)
  if (is_int_dtype(a$dtype) && all(abs(d) < 2^31)) storage.mode(d) <- "integer"
  if (length(a$shape) >= 3L) print_c_order(d, a$shape, digits) else print(d, digits = digits)
  invisible(x)
}

# Rank >= 3 arrays print as 0-based indexed matrices.
print_c_order <- function(d, shape, digits = NULL) {
  r <- length(shape)
  lead <- shape[seq_len(r - 2L)]
  if (prod(lead) == 0) {
    cat("<empty>\n")
    return(invisible())
  }
  grid <- as.matrix(expand.grid(lapply(rev(lead), function(n) seq_len(n) - 1L)))
  grid <- grid[, rev(seq_len(ncol(grid))), drop = FALSE]
  for (k in seq_len(nrow(grid))) {
    idx <- grid[k, ]
    cat("[", paste(idx, collapse = ", "), ", , ]\n", sep = "")
    m <- do.call(`[`, c(list(d), as.list(idx + 1L), list(TRUE, TRUE, drop = FALSE)))
    dim(m) <- shape[(r - 1L):r]
    print(m, digits = digits)
  }
}
