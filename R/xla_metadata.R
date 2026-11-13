# Port of jax.experimental.xla_metadata (jax/_src/xla_metadata.py): XLA frontend
# attributes on the operations of a compiled program.

#' Attach XLA metadata to operations
#'
#' Ports of `jax.experimental.xla_metadata`. XLA metadata are frontend attributes
#' (string key value pairs) on the lowered operations, which XLA reads as hints, for
#' example `FUSE_LIMIT` (a fusion boundary) or `MUST_FUSE` (fuse a group).
#'
#' * `experimental_xla_metadata_set_xla_metadata(x, ...)` tags the operations that
#'   produce the arrays of the pytree `x` and returns `x`.
#' * `experimental_xla_metadata_set_xla_metadata(..., code = )` evaluates `code` with
#'   the metadata in effect: every operation staged meanwhile (under [jit()] and the
#'   other transformations that compile) is tagged. Nested uses merge, the inner value
#'   of a key winning; a value of `NULL` removes a key. Without `x` and `code` it
#'   returns a function that wraps a function to run with the metadata in effect (the
#'   decorator form of JAX).
#' * `experimental_xla_metadata_fuse_limit(code)` evaluates `code` with
#'   `FUSE_LIMIT = TRUE`.
#' * `experimental_xla_metadata_must_fuse_call(identifier)` returns a decorator that
#'   tags the operations of a function with `MUST_FUSE = identifier`.
#' * `experimental_xla_metadata_xla_metadata_call(f, ...)` and
#'   `experimental_xla_metadata_xla_metadata_call2(f, metadata, ad_metadata)` wrap `f`
#'   so that its operations carry the metadata.
#'
#' Values may be strings, logicals or numbers; they are attached as strings (logicals
#' as `"true"` and `"false"`, doubles as Python writes floats). Unlike JAX, which
#' stages `xla_metadata_call` as a call operation tagged as a whole, jaxR tags the
#' operations of the function directly, which XLA treats the same once it inlines the
#' call; and the operations of the backward pass of [grad()] are tagged only when they
#' are staged inside the context (`ad_metadata` other than `"drop"` is not applied).
#'
#' @param x A pytree of arrays to tag, or `NULL`.
#' @param ... Metadata as named arguments.
#' @param code An expression evaluated with the metadata in effect.
#' @return The tagged `x`, the value of `code`, or a function.
#' @examples
#' \dontrun{
#' f <- jit(function(x) {
#'   y <- experimental_xla_metadata_set_xla_metadata(jnp_sin(x), my_key = "a")
#'   experimental_xla_metadata_fuse_limit(y * 2)
#' })
#' cat(f$lower(jnp_ones(3))$as_text())
#' }
#' @export
experimental_xla_metadata_set_xla_metadata <- function(x = NULL, ..., code) {
  meta <- xm_meta(list(...))
  if (!missing(code)) return(xm_with(meta, code))
  if (is.null(x)) {
    return(function(f) {
      force(f)
      function(...) xm_with(meta, f(...))
    })
  }
  if (!length(meta)) return(x)
  tree_map(function(v) bind(xla_metadata_value_p, list(as_jax(v)), list(xla_metadata_kvs = meta)), x)
}

#' @rdname experimental_xla_metadata_set_xla_metadata
#' @export
experimental_xla_metadata_fuse_limit <- function(code) xm_with(xm_meta(list(FUSE_LIMIT = TRUE)), code)

#' @rdname experimental_xla_metadata_set_xla_metadata
#' @param identifier The fusion group.
#' @export
experimental_xla_metadata_must_fuse_call <- function(identifier) {
  experimental_xla_metadata_xla_metadata_call(MUST_FUSE = identifier)
}

#' @rdname experimental_xla_metadata_set_xla_metadata
#' @param f A function, or `NULL` for a decorator.
#' @export
experimental_xla_metadata_xla_metadata_call <- function(f = NULL, ...) {
  experimental_xla_metadata_xla_metadata_call2(f, list(...))
}

#' @rdname experimental_xla_metadata_set_xla_metadata
#' @param metadata A named list of metadata.
#' @param ad_metadata `"same"` (the default), `"drop"`, or a named list: the metadata
#'   of the computations autodiff derives from `f` (see Details).
#' @export
experimental_xla_metadata_xla_metadata_call2 <- function(f = NULL, metadata = NULL, ad_metadata = "same") {
  if (is.list(f) && is.null(metadata)) { metadata <- f; f <- NULL }
  if (!is.null(f) && !is.function(f))
    jax_error("expected a callable to wrap, got ", class(f)[1L], class = "jax_type_error")
  meta <- xm_meta(metadata %||% list())
  wrap <- function(g) {
    force(g)
    function(...) xm_with(meta, g(...))
  }
  if (is.null(f)) wrap else wrap(f)
}

# Port of _canonicalize_metadata and the string conversion of the lowering.
xm_meta <- function(meta) {
  if (length(meta) && (is.null(names(meta)) || !all(nzchar(names(meta)))))
    jax_error("XLA metadata must be given as named arguments", class = "jax_type_error")
  lapply(meta, function(v) {
    if (is.null(v)) return(NULL)
    if (!is.atomic(v) || length(v) != 1L || is.na(v))
      jax_error("XLA metadata values must be strings, logicals or numbers, got ",
                class(v)[1L], class = "jax_type_error")
    if (is.logical(v)) return(if (v) "true" else "false")
    if (is.double(v)) return(if (is.finite(v) && v == trunc(v) && abs(v) < 1e16)
      sprintf("%.1f", v) else as.character(v))
    as.character(v)
  })
}

# Port of update_metadata: merge, the new values winning, NULL removing a key.
xm_with <- function(meta, code) {
  old <- .jx$xla_metadata
  new <- old %||% list()
  for (k in names(meta)) new[k] <- list(meta[[k]])
  new <- Filter(Negate(is.null), new)
  .jx$xla_metadata <- if (length(new)) new[order(names(new))]
  on.exit(.jx$xla_metadata <- old, add = TRUE)
  code
}

# ---- lowering -------------------------------------------------------------------------

# Port of _update_frontend_attributes on the MLIR text: the operation defining the
# SSA value `v` among lines `from` to ctx$n gets the attributes, merged with those it
# has (existing values win when `keep_existing`).
xm_tag_value <- function(ctx, v, attrs, from = 1L, keep_existing = FALSE) {
  name <- sub("#.*$", "", v)
  pat <- paste0("^\\s*", gsub("([%])", "\\1", name), "(:[0-9]+)? = ")
  for (i in rev(seq.int(from, length.out = max(0L, ctx$n - from + 1L)))) {
    line <- ctx$lines[[i]]
    if (grepl(pat, line)) {
      ctx$lines[[i]] <- xm_add_attrs(line, attrs, keep_existing)
      return(invisible(TRUE))
    }
  }
  invisible(FALSE)
}

# Insert or merge mhlo.frontend_attributes in the discardable attribute dictionary of
# a generic form operation line. The line is scanned with string literals skipped, so
# that braces, commas or " : (" inside attribute strings are left alone.
xm_add_attrs <- function(line, attrs, keep_existing) {
  ch <- strsplit(line, "")[[1L]]
  n <- length(ch)
  # Positions outside string literals, and the depth of brackets there.
  out <- logical(n)
  inq <- FALSE
  i <- 1L
  while (i <= n) {
    if (inq) {
      if (ch[i] == "\\") i <- i + 1L else if (ch[i] == "\"") inq <- FALSE
    } else if (ch[i] == "\"") inq <- TRUE else out[i] <- TRUE
    i <- i + 1L
  }
  arrow <- ch == ">" & c("", ch[-n]) == "-"
  depth <- cumsum(ifelse(out & ch %in% c("{", "(", "[", "<"), 1L,
                         ifelse(out & ch %in% c("}", ")", "]", ">") & !arrow, -1L, 0L)))
  # The operation signature: the last " : (" at depth zero.
  starts <- which(out & ch == ":" & c(ch[-1L], "") == " " & c("", ch[-n]) == " " &
                    c(ch[-(1:2)], "", "") == "(" & depth == 0L)
  if (!length(starts)) return(line)
  sig <- max(starts) - 1L
  head <- substr(line, 1L, sig - 1L)
  tail <- substr(line, sig, n)
  m <- sig - 1L
  # The trailing attribute dictionary: a "}" closing a "{" that follows a space.
  open <- NA_integer_
  if (m >= 1L && ch[m] == "}" && out[m]) {
    cand <- which(out[seq_len(m)] & ch[seq_len(m)] == "{" & c(0L, depth)[seq_len(m)] == depth[m])
    cand <- cand[cand < m]
    if (length(cand)) open <- max(cand)
    if (!is.na(open) && open > 1L && ch[open - 1L] != " ") open <- NA_integer_
  }
  if (is.na(open)) return(paste0(head, " {mhlo.frontend_attributes = ", xm_dict(attrs), "}", tail))
  entries <- xm_split(substr(line, open + 1L, m - 1L))
  key <- sub("^\\s*([^ =]+).*$", "\\1", entries)
  hit <- which(key == "mhlo.frontend_attributes")
  if (length(hit)) {
    ex <- xm_parse_dict(sub("^[^=]*=\\s*", "", entries[hit[1L]]))
    merged <- if (keep_existing) c(attrs[setdiff(names(attrs), names(ex))], ex)
              else c(ex[setdiff(names(ex), names(attrs))], attrs)
    entries[hit[1L]] <- paste0("mhlo.frontend_attributes = ", xm_dict(merged))
  } else {
    entries <- c(entries, paste0("mhlo.frontend_attributes = ", xm_dict(attrs)))
  }
  paste0(substr(line, 1L, open), paste(trimws(entries), collapse = ", "), "}", tail)
}

# A dictionary of string attributes, as the lowering of JAX writes it.
xm_dict <- function(a) ffi_dict_attr(lapply(a, mlir_string))

# Split the text at its commas outside string literals and brackets.
xm_split <- function(txt) {
  ch <- strsplit(txt, "")[[1L]]
  if (!length(ch)) return(character())
  parts <- character()
  cur <- 1L
  d <- 0L
  inq <- FALSE
  i <- 1L
  while (i <= length(ch)) {
    c1 <- ch[i]
    if (inq) {
      if (c1 == "\\") i <- i + 1L else if (c1 == "\"") inq <- FALSE
    } else if (c1 == "\"") inq <- TRUE
    else if (c1 %in% c("{", "(", "[", "<")) d <- d + 1L
    else if (c1 %in% c("}", ")", "]", ">")) d <- d - 1L
    else if (c1 == "," && d == 0L) {
      parts <- c(parts, substr(txt, cur, i - 1L))
      cur <- i + 1L
    }
    i <- i + 1L
  }
  parts <- c(parts, substr(txt, cur, length(ch)))
  parts[nzchar(trimws(parts))]
}

# The string entries of a dictionary attribute "{k = "v", "a b" = "w"}", decoded.
xm_parse_dict <- function(txt) {
  txt <- trimws(txt)
  inner <- substr(txt, 2L, nchar(txt) - 1L)
  out <- list()
  for (e in xm_split(inner)) {
    e <- trimws(e)
    if (startsWith(e, "\"")) {
      kend <- xm_string_end(e)
      k <- xm_unstring(substr(e, 1L, kend))
      rest <- substr(e, kend + 1L, nchar(e))
    } else {
      k <- sub("\\s*=.*$", "", e)
      rest <- substr(e, nchar(k) + 1L, nchar(e))
    }
    v <- trimws(sub("^\\s*=", "", rest))
    out[[k]] <- if (startsWith(v, "\"")) xm_unstring(v) else v
  }
  out
}

# The position of the quote closing the string literal that starts the text.
xm_string_end <- function(txt) {
  ch <- strsplit(txt, "")[[1L]]
  i <- 2L
  while (i <= length(ch)) {
    if (ch[i] == "\\") i <- i + 1L else if (ch[i] == "\"") return(i)
    i <- i + 1L
  }
  length(ch)
}

# Decode an MLIR string literal (\\, \", \n, \t and \XX escapes).
xm_unstring <- function(lit) {
  b <- charToRaw(lit)
  b <- b[-c(1L, length(b))]
  out <- raw(0)
  i <- 1L
  while (i <= length(b)) {
    if (b[i] == charToRaw("\\") && i < length(b)) {
      nx <- rawToChar(b[i + 1L])
      hex <- if (i + 2L <= length(b)) rawToChar(b[(i + 1L):(i + 2L)]) else ""
      if (grepl("^[0-9A-Fa-f]{2}$", hex)) {
        out <- c(out, as.raw(strtoi(hex, 16L)))
        i <- i + 3L
        next
      }
      out <- c(out, charToRaw(switch(nx, n = "\n", t = "\t", nx)))
      i <- i + 2L
      next
    }
    out <- c(out, b[i])
    i <- i + 1L
  }
  r <- rawToChar(out)
  Encoding(r) <- "UTF-8"
  r
}

# Port of xla_metadata_value_p: an identity that tags the operation producing its operand.
xla_metadata_value_p <- new_primitive("xla_metadata_value")
xla_metadata_value_p$abstract_eval <- function(avals, params) avals[[1L]]
xla_metadata_value_p$impl <- function(args, params) args[[1L]]
xla_metadata_value_p$jvp <- function(primals, tangents, params) {
  list(bind(xla_metadata_value_p, primals, params), tangents[[1L]])
}
xla_metadata_value_p$transpose <- function(ct, args, params) list(ct)
xla_metadata_value_p$batch <- function(vals, dims, params, axis) {
  list(bind(xla_metadata_value_p, vals, params), dims[[1L]])
}
xla_metadata_value_p$lowering <- function(ctx, ins, avals_in, avals_out, params) {
  xm_tag_value(ctx, ins[[1L]], params$xla_metadata_kvs, keep_existing = TRUE)
  ins
}
