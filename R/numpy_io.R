# NumPy file I/O, array printing and index helpers re-exported by jax.numpy
# (jnp.load of jax/_src/numpy/lax_numpy.py; numpy's save, savez, array_repr, array_str,
# s_ and index_exp, which jax.numpy re-exports from numpy).

# ---- .npy format ----------------------------------------------------------------------

npy_descr <- c(bool = "|b1", int8 = "|i1", int16 = "<i2", int32 = "<i4", int64 = "<i8",
               uint8 = "|u1", uint16 = "<u2", uint32 = "<u4", uint64 = "<u8",
               float16 = "<f2", float32 = "<f4", float64 = "<f8", complex64 = "<c8",
               complex128 = "<c16", bfloat16 = "<V2")

npy_dtype_of_descr <- function(descr) {
  d <- sub("^[<|=]", "", descr)
  big <- substr(descr, 1L, 1L) == ">"
  d <- sub("^>", "", d)
  dt <- switch(d, b1 = "bool", i1 = "int8", i2 = "int16", i4 = "int32", i8 = "int64",
               u1 = "uint8", u2 = "uint16", u4 = "uint32", u8 = "uint64", f2 = "float16",
               f4 = "float32", f8 = "float64", c8 = "complex64", c16 = "complex128",
               V2 = "bfloat16", NA_character_)
  if (is.na(dt)) jax_error("unsupported .npy dtype '", descr, "'", class = "jax_value_error")
  list(dtype = dt, big_endian = big)
}

# Raw little endian bytes of an array, in C order, computed on the device.
npy_bytes <- function(x) {
  dt <- aval_of(x)$dtype
  if (dt == "bool") x <- lax_convert_element_type(x, "uint8")
  if (is_complex_dtype(dt)) x <- jnp_stack(list(jnp_real(x), jnp_imag(x)), axis = -1L)
  if (aval_of(x)$dtype != "uint8") x <- lax_bitcast_convert_type(x, "uint8")
  as.raw(c_ravel(as.array(x)))
}

npy_from_bytes <- function(bytes, dtype, shape, big_endian) {
  nbytes <- switch(dtype, bool = 1L, complex64 = 4L, complex128 = 8L, dtype_bits(dtype) %/% 8L)
  if (big_endian && nbytes > 1L) {
    m <- matrix(bytes, nrow = nbytes)
    bytes <- as.vector(m[nbytes:1L, , drop = FALSE])
  }
  n <- prod_int(shape)
  narrow <- canonicalize_dtype(dtype)
  if (narrow != dtype) {
    # x64 is disabled: the stored 64 bit values are narrowed as NumPy's astype does
    # (integers keep their low half, floats round).
    if (is_int_dtype(dtype)) {
      low <- as.vector(matrix(bytes, nrow = 8L)[1:4, , drop = FALSE])
      return(npy_from_bytes(low, narrow, shape, FALSE))
    }
    v <- readBin(bytes, "double", n = length(bytes) %/% 8L, size = 8L, endian = "little")
    if (is_complex_dtype(dtype)) {
      re <- seq(1L, by = 2L, length.out = length(v) %/% 2L)
      v <- complex(real = v[re], imaginary = v[re + 1L])
    }
    return(jnp_reshape(as_jax(array(v, dim = length(v)), narrow), shape))
  }
  u8 <- as_jax(array(as.integer(bytes), dim = length(bytes)), "uint8")
  if (dtype == "bool") return(jnp_reshape(lax_ne(u8, lax_full_like(u8, 0)), shape))
  if (is_complex_dtype(dtype)) {
    part <- complex_component(dtype)
    f <- lax_bitcast_convert_type(jnp_reshape(u8, c(n, 2L, nbytes)), part)
    return(jnp_reshape(lax_complex(f[, 0L], f[, 1L]), shape))
  }
  if (nbytes == 1L) return(jnp_reshape(lax_bitcast_convert_type(u8, dtype), shape))
  jnp_reshape(lax_bitcast_convert_type(jnp_reshape(u8, c(n, nbytes)), dtype), shape)
}

npy_write <- function(con, arr) {
  arr <- as_jax(arr)
  a <- aval_of(arr)
  descr <- npy_descr[a$dtype]
  if (is.na(descr)) jax_error("cannot save arrays of dtype ", a$dtype, " to .npy", class = "jax_value_error")
  shape <- a$shape
  shape_str <- if (!length(shape)) "()" else if (length(shape) == 1L) paste0("(", shape, ",)")
               else paste0("(", paste(shape, collapse = ", "), ")")
  header <- paste0("{'descr': '", descr, "', 'fortran_order': False, 'shape': ", shape_str, ", }")
  total <- 10L + nchar(header) + 1L
  pad <- (64L - total %% 64L) %% 64L
  header <- paste0(header, strrep(" ", pad), "\n")
  writeBin(as.raw(c(0x93, charToRaw("NUMPY"), 1L, 0L)), con)
  writeBin(as.integer(nchar(header)), con, size = 2L, endian = "little")
  writeBin(charToRaw(header), con)
  if (prod_int(shape)) writeBin(npy_bytes(arr), con)
}

npy_read <- function(con) {
  magic <- readBin(con, "raw", 6L)
  if (!identical(magic, as.raw(c(0x93, charToRaw("NUMPY")))))
    jax_error("not a .npy file", class = "jax_value_error")
  ver <- as.integer(readBin(con, "raw", 2L))
  hlen <- if (ver[1L] == 1L) readBin(con, "integer", 1L, size = 2L, signed = FALSE, endian = "little")
          else readBin(con, "integer", 1L, size = 4L, endian = "little")
  header <- rawToChar(readBin(con, "raw", hlen))
  descr <- sub(".*'descr': *'([^']*)'.*", "\\1", header)
  fortran <- grepl("'fortran_order': *True", header)
  shp <- sub(".*'shape': *\\(([^)]*)\\).*", "\\1", header)
  shape <- as.integer(Filter(nzchar, trimws(strsplit(shp, ",")[[1L]])))
  info <- npy_dtype_of_descr(descr)
  nbytes <- switch(info$dtype, bool = 1L, dtype_bits(info$dtype) %/% 8L)
  n <- prod_int(shape)
  bytes <- readBin(con, "raw", n * nbytes)
  if (!n) return(jnp_zeros(shape, info$dtype))
  if (fortran) {
    out <- npy_from_bytes(bytes, info$dtype, rev(shape), info$big_endian)
    return(jnp_transpose(out))
  }
  npy_from_bytes(bytes, info$dtype, shape, info$big_endian)
}

#' Save and load arrays in NumPy formats
#'
#' Ports of `jnp.save`, `jnp.savez` and `jnp.load` (NumPy's `.npy` and `.npz`
#' formats, readable by NumPy). `bfloat16` arrays are stored as two byte voids, as with
#' NumPy, and restored by `jnp_load()`. `.npz` files load as a named list of arrays.
#' @param file A file name or connection. `jnp_save()` adds `.npy` and `jnp_savez()`
#'   `.npz` to file names without the extension.
#' @param arr An array.
#' @param ... Arrays for `jnp_savez()` (named ones keep their names, others are
#'   `arr_0`, `arr_1`, ...); ignored by `jnp_load()`.
#' @param allow_pickle,fix_imports,mmap_mode,encoding Accepted for compatibility.
#' @return `jnp_load()` returns an array or a named list of arrays; the others return
#'   `NULL` invisibly.
#' @export
jnp_save <- function(file, arr, allow_pickle = TRUE, fix_imports = TRUE) {
  if (is.character(file)) {
    if (!grepl("\\.npy$", file)) file <- paste0(file, ".npy")
    con <- file(file, "wb")
    on.exit(close(con), add = TRUE)
  } else con <- file
  npy_write(con, arr)
  invisible(NULL)
}

npz_empty <- as.raw(c(0x50, 0x4b, 0x05, 0x06, rep(0, 18L)))

#' @rdname jnp_save
#' @export
jnp_savez <- function(file, ...) {
  arrays <- list(...)
  nm <- names(arrays) %||% rep("", length(arrays))
  unnamed <- !nzchar(nm)
  auto <- paste0("arr_", seq_len(sum(unnamed)) - 1L)
  clash <- intersect(auto, nm[!unnamed])
  if (length(clash))
    jax_error("Cannot use un-named variables and keyword ", clash[1L], class = "jax_value_error")
  nm[unnamed] <- auto
  if (is.character(file) && !grepl("\\.npz$", file)) file <- paste0(file, ".npz")
  if (!length(arrays)) {
    # zip refuses to make an archive without members: write the empty archive (its end
    # of central directory record) directly, as NumPy does.
    writeBin(npz_empty, file)
    return(invisible(NULL))
  }
  dir <- tempfile("npz")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  files <- file.path(dir, paste0(nm, ".npy"))
  for (i in seq_along(arrays)) jnp_save(files[i], arrays[[i]])
  # The archive is written from inside a temporary directory, so the target must be
  # absolute even when it does not exist yet.
  target <- if (is.character(file))
    file.path(normalizePath(dirname(file), mustWork = TRUE), basename(file)) else file
  unlink(target)
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)
  status <- utils::zip(target, basename(files), flags = "-q0X")
  if (!identical(as.integer(status), 0L)) jax_error("zip failed while writing ", file)
  invisible(NULL)
}

#' @rdname jnp_save
#' @export
jnp_load <- function(file, mmap_mode = NULL, allow_pickle = FALSE, fix_imports = TRUE,
                     encoding = "ASCII", ...) {
  if (is.character(file)) {
    con <- file(file, "rb")
    magic <- readBin(con, "raw", 4L)
    close(con)
    if (identical(magic, npz_empty[1:4])) return(stats::setNames(list(), character(0)))
    if (identical(magic, as.raw(c(0x50, 0x4b, 0x03, 0x04)))) {
      dir <- tempfile("npz")
      on.exit(unlink(dir, recursive = TRUE), add = TRUE)
      members <- utils::unzip(file, exdir = dir)
      out <- lapply(members, jnp_load)
      names(out) <- sub("\\.npy$", "", basename(members))
      return(out)
    }
    con <- file(file, "rb")
    on.exit(close(con), add = TRUE)
  } else con <- file
  npy_read(con)
}

# ---- array_repr / array_str ---------------------------------------------------------------

# Fewest significant digits that round trip to the same value of `dtype` (NumPy's
# unique float formatting).
np_shortest_digits <- function(v, dtype) {
  maxd <- if (dtype == "float64") 17L else 9L
  need <- rep(maxd, length(v))
  left <- seq_along(v)
  for (d in seq_len(maxd)) {
    if (!length(left)) break
    cand <- signif(v[left], d)
    back <- if (dtype == "float64") cand else as.vector(as.array(as_jax(cand, dtype)))
    ok <- back == v[left]
    need[left[ok]] <- d
    left <- left[!ok]
  }
  need
}

np_fmt_float <- function(v, dtype, precision = 8L, suppress_small = FALSE, sign = FALSE) {
  out <- character(length(v))
  out[is.nan(v)] <- "nan"
  out[is.infinite(v)] <- ifelse(v[is.infinite(v)] > 0, "inf", "-inf")
  fin <- which(is.finite(v))
  if (sign) out[!is.finite(v) & !is.nan(v) & v > 0] <- "+inf"
  if (sign) out[is.nan(v)] <- "+nan"
  if (!length(fin)) return(out)
  x <- v[fin]
  nz <- abs(x[x != 0])
  sci <- length(nz) && !suppress_small && (max(nz) >= 1e8 || min(nz) < 1e-4 || max(nz) / min(nz) > 1e3)
  need <- ifelse(x == 0, 1L, np_shortest_digits(x, dtype))
  if (sci) {
    mant <- min(max(need - 1L), precision)
    s <- formatC(x, digits = mant, format = "e")
    s <- sub("e([+-])(\\d)$", "e\\10\\2", s)
    if (mant == 0L) s <- sub("e", ".e", s)
    if (sign) s <- ifelse(x >= 0, paste0("+", s), s)
    out[fin] <- s
    return(out)
  }
  xr <- signif(x, need)
  mag <- ifelse(xr == 0, 0, floor(log10(abs(xr))))
  frac <- pmin(pmax(need - mag - 1, 0), precision)
  fd <- max(frac)
  s <- vapply(seq_along(x), function(i) {
    t <- formatC(xr[i], digits = frac[i], format = "f")
    if (frac[i] == 0) t <- paste0(t, ".")
    paste0(t, strrep(" ", fd - frac[i]))
  }, "")
  if (sign) s <- ifelse(x >= 0 & !grepl("^-", s), paste0("+", s), s)
  out[fin] <- s
  out
}

np_fmt_values <- function(v, dtype, precision = 8L, suppress_small = FALSE) {
  if (dtype == "bool") return(ifelse(v, "True", "False"))
  if (is_int_dtype(dtype)) return(format(v, scientific = FALSE, trim = TRUE))
  if (is_complex_dtype(dtype)) {
    part <- complex_component(dtype)
    re <- np_fmt_float(Re(v), part, precision, suppress_small)
    im <- np_fmt_float(Im(v), part, precision, suppress_small, sign = TRUE)
    re <- formatC(re, width = max(nchar(re)), flag = " ")
    im <- formatC(im, width = max(nchar(im)), flag = "-")
    return(paste0(re, sub("( *)$", "j\\1", im)))
  }
  np_fmt_float(v, dtype, precision, suppress_small)
}

np_format_array <- function(x, sep, prefix, max_line_width = 75L, precision = 8L, suppress_small = FALSE) {
  a <- aval_of(x)
  shape <- a$shape
  v <- c_ravel(array_host(x))
  strs <- np_fmt_values(v, a$dtype, precision, suppress_small)
  width <- if (length(strs)) max(nchar(strs)) else 0L
  strs <- formatC(strs, width = width)
  if (!length(shape)) return(trimws(strs))
  if (!prod_int(shape)) return(paste0(strrep("[", length(shape)), strrep("]", length(shape))))
  rsep <- sub(" +$", "", sep)
  rec <- function(dim, offset, indent) {
    n <- shape[dim]
    stride <- prod_int(shape[seq_along(shape) > dim])
    if (dim == length(shape)) {
      line <- paste0(indent, "[")
      lines <- character(0)
      for (k in seq_len(n)) {
        word <- paste0(strs[offset + k], if (k < n) sep else "")
        if (nchar(line) + nchar(sub(" +$", "", word)) > max_line_width - (length(shape) - dim) && k > 1L) {
          lines <- c(lines, sub(" +$", "", line))
          line <- paste0(indent, " ", word)
        } else line <- paste0(line, word)
      }
      lines <- c(lines, paste0(line, "]"))
      return(paste(lines, collapse = "\n"))
    }
    parts <- vapply(seq_len(n) - 1L, function(i) rec(dim + 1L, offset + i * stride, paste0(indent, " ")), "")
    parts[1L] <- sub(paste0("^", indent, " "), "", parts[1L])
    blank <- strrep("\n", length(shape) - dim)
    paste0(indent, "[", paste(parts, collapse = paste0(rsep, blank)), "]")
  }
  out <- rec(1L, 0L, strrep(" ", nchar(prefix)))
  sub(paste0("^", strrep(" ", nchar(prefix))), "", out)
}

#' NumPy style string representations
#'
#' Ports of `jnp.array_repr` and `jnp.array_str` (NumPy's functions): `"array([1., 2.],
#' dtype=float32)"` and `"[1. 2.]"`.
#' @param arr An array.
#' @param max_line_width Maximum line width.
#' @param precision Number of digits.
#' @param suppress_small Print very small numbers as zero.
#' @return A string.
#' @export
jnp_array_repr <- function(arr, max_line_width = NULL, precision = NULL, suppress_small = NULL) {
  x <- as_jax(arr)
  dt <- aval_of(x)$dtype
  body <- np_format_array(x, ", ", "array(", max_line_width %||% 75L, precision %||% 8L, isTRUE(suppress_small))
  implicit <- dt %in% c("bool", "int64", "float64", "complex128") && prod_int(aval_of(x)$shape) > 0L
  paste0("array(", body, if (!implicit) paste0(", dtype=", dt), ")")
}

#' @rdname jnp_array_repr
#' @export
jnp_array_str <- function(arr, max_line_width = NULL, precision = NULL, suppress_small = NULL) {
  np_format_array(as_jax(arr), " ", "", max_line_width %||% 75L, precision %||% 8L, isTRUE(suppress_small))
}

# ---- index helpers ------------------------------------------------------------------------

#' Index expression builders
#'
#' Ports of `jnp.s_` and `jnp.index_exp` (NumPy's): `jnp_s_[...]` returns the index
#' written between the brackets (one index as is, several as an index tuple that `x[...]`
#' accepts), and `jnp_index_exp[...]` always returns an index tuple. Indices are 0-based
#' and missing ones are full slices, as in `x[...]`.
#' @format Index builder objects.
#' @examples
#' \dontrun{
#' x <- jnp_reshape(jnp_arange(12), c(3L, 4L))
#' x[jnp_s_[1L, sl(0L, 2L)]]
#' }
#' @name jnp_s_
NULL

#' @rdname jnp_s_
#' @export
jnp_s_ <- structure(list(maketuple = FALSE), class = "jax_index_maker")

#' @rdname jnp_s_
#' @export
jnp_index_exp <- structure(list(maketuple = TRUE), class = "jax_index_maker")

#' @export
`[.jax_index_maker` <- function(x, ...) {
  idx <- idx_collect(...)
  if (length(idx) == 1L && !.subset2(x, "maketuple")) return(idx[[1L]])
  structure(idx, class = "jax_index_tuple")
}

#' @export
print.jax_index_maker <- function(x, ...) {
  cat(if (.subset2(x, "maketuple")) "<index_exp>\n" else "<s_>\n")
  invisible(x)
}

#' NumPy warning classes
#'
#' `jnp_ComplexWarning` is the port of `jnp.ComplexWarning` (NumPy's
#' `ComplexWarning`): the class of the warnings jaxR signals when a complex value is
#' cast to a real type.
#' @format A character scalar naming a condition class.
#' @export
jnp_ComplexWarning <- "jax_complex_warning"
