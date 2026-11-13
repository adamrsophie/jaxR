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
# a generic form operation line.
xm_add_attrs <- function(line, attrs, keep_existing) {
  sig <- regexpr(" : \\([^\n]*$", line)
  if (sig < 0L) return(line)
  head <- substr(line, 1L, sig - 1L)
  tail <- substr(line, sig, nchar(line))
  chars <- strsplit(head, "")[[1L]]
  fmt <- function(a) paste0("{", paste0(names(a), " = \"", gsub("([\"\\\\])", "\\\\\\1", unlist(a)), "\"",
                                        collapse = ", "), "}")
  if (length(chars) && chars[length(chars)] == "}") {
    # Find the opening brace of the trailing dictionary, skipping quoted strings.
    depth <- 0L
    inq <- FALSE
    open <- NA_integer_
    for (i in rev(seq_along(chars))) {
      ch <- chars[i]
      if (ch == "\"" && (i == 1L || chars[i - 1L] != "\\")) inq <- !inq
      if (inq) next
      if (ch == "}") depth <- depth + 1L
      else if (ch == "{") { depth <- depth - 1L; if (depth == 0L) { open <- i; break } }
    }
    if (!is.na(open) && (open == 1L || chars[open - 1L] == " ") &&
        !(open > 1L && chars[open - 1L] == "<")) {
      dict <- paste(chars[(open + 1L):(length(chars) - 1L)], collapse = "")
      m <- regexpr("mhlo\\.frontend_attributes = \\{[^}]*\\}", dict)
      if (m > 0L) {
        old <- regmatches(dict, m)
        kv <- regmatches(old, gregexpr("([A-Za-z_][A-Za-z0-9_.$]*|\"[^\"]*\") = \"([^\"\\\\]|\\\\.)*\"", old))[[1L]]
        ex <- list()
        for (p in kv) {
          k <- sub(" = .*$", "", p)
          ex[[k]] <- sub("\"$", "", sub("^[^=]*= \"", "", p))
        }
        merged <- if (keep_existing) c(attrs[setdiff(names(attrs), names(ex))], ex)
                  else c(ex[setdiff(names(ex), names(attrs))], attrs)
        merged <- lapply(merged, function(s) gsub("\\\\(.)", "\\1", s))
        dict <- sub("mhlo\\.frontend_attributes = \\{[^}]*\\}",
                    paste0("mhlo.frontend_attributes = ", gsub("\\\\", "\\\\\\\\", fmt(merged))), dict)
      } else {
        dict <- paste0(dict, ", mhlo.frontend_attributes = ", fmt(attrs))
      }
      return(paste0(paste(chars[seq_len(open - 1L)], collapse = ""), "{", dict, "}", tail))
    }
  }
  paste0(head, " {mhlo.frontend_attributes = ", fmt(attrs), "}", tail)
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
