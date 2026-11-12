# Base R generics for jax values: arithmetic operators, math functions, summaries.

#' Operators and math functions for jax arrays
#'
#' Arithmetic (`+ - * / ^ %% %/%`), comparison, logical (`& | !`) operators and
#' `%*%` follow `jax.numpy` semantics (broadcasting, type promotion). Math group
#' functions (`exp`, `log`, `sqrt`, `abs`, ...) and summaries (`sum`, `max`, ...)
#' map to the `jnp_` functions. These work on tracers too, so ordinary R code is
#' transformable by [grad()], [vmap()] and [jit()].
#' @param e1,e2,x Jax values or R data.
#' @param ... Further arguments.
#' @param na.rm Ignored.
#' @name jax_operators
NULL

#' @rdname jax_operators
#' @export
Ops.jax_value <- function(e1, e2) {
  if (inherits(e1, "jax_sparse") || (!missing(e2) && inherits(e2, "jax_sparse")))
    return(sp_ops(.Generic, e1, if (missing(e2)) NULL else e2, missing(e2)))
  if (missing(e2)) {
    return(switch(.Generic,
      "-" = jnp_negative(e1),
      "+" = as_jax(e1),
      "!" = {
        x <- as_jax(e1)
        if (aval_of(x)$dtype == "bool") lax_bitwise_not(x) else jnp_logical_not(x)
      },
      jax_error("unsupported unary operator ", .Generic)))
  }
  if (inherits(e1, "sym")) e1 <- sym_to_jax_constant(e1)
  if (inherits(e2, "sym")) e2 <- sym_to_jax_constant(e2)
  switch(.Generic,
    "+" = jnp_add(e1, e2),
    "-" = jnp_subtract(e1, e2),
    "*" = jnp_multiply(e1, e2),
    "/" = jnp_true_divide(e1, e2),
    "^" = jnp_power(e1, e2),
    "%%" = jnp_remainder(e1, e2),
    "%/%" = jnp_floor_divide(e1, e2),
    "==" = jnp_equal(e1, e2),
    "!=" = jnp_not_equal(e1, e2),
    "<" = jnp_less(e1, e2),
    "<=" = jnp_less_equal(e1, e2),
    ">" = jnp_greater(e1, e2),
    ">=" = jnp_greater_equal(e1, e2),
    "&" = jnp_bitwise_and(e1, e2),
    "|" = jnp_bitwise_or(e1, e2),
    jax_error("unsupported operator ", .Generic))
}

#' @rdname jax_operators
#' @export
matrixOps.jax_value <- function(x, y) {
  if (inherits(x, "jax_sparse") || inherits(y, "jax_sparse")) return(sp_matmul(x, y))
  switch(.Generic, "%*%" = jnp_matmul(x, y), jax_error("unsupported operator ", .Generic))
}

#' @rdname jax_operators
#' @param y Second operand.
#' @param mx,my,cl,reverse See [base::chooseOpsMethod()].
#' @export
chooseOpsMethod.jax_value <- function(x, y, mx, my, cl, reverse) TRUE

#' @rdname jax_operators
#' @export
Math.jax_value <- function(x, ...) {
  switch(.Generic,
    abs = jnp_abs(x), sign = jnp_sign(x), sqrt = jnp_sqrt(x), floor = jnp_floor(x),
    ceiling = jnp_ceil(x), trunc = jnp_trunc(x),
    round = {
      d <- list(...)
      jnp_round(x, if (length(d)) as.integer(d[[1L]]) else 0L)
    },
    signif = jax_error("signif is not supported for jax arrays"),
    exp = jnp_exp(x), expm1 = jnp_expm1(x), log1p = jnp_log1p(x), log2 = jnp_log2(x),
    log10 = jnp_log10(x),
    log = {
      dots <- list(...)
      if (length(dots)) jnp_true_divide(jnp_log(x), jnp_log(dots[[1L]])) else jnp_log(x)
    },
    cos = jnp_cos(x), sin = jnp_sin(x), tan = jnp_tan(x),
    cospi = jnp_cos(jnp_multiply(x, pi)), sinpi = jnp_sin(jnp_multiply(x, pi)),
    tanpi = jnp_tan(jnp_multiply(x, pi)),
    acos = jnp_arccos(x), asin = jnp_arcsin(x), atan = jnp_arctan(x),
    cosh = jnp_cosh(x), sinh = jnp_sinh(x), tanh = jnp_tanh(x),
    acosh = jnp_arccosh(x), asinh = jnp_arcsinh(x), atanh = jnp_arctanh(x),
    lgamma = scipy_special_gammaln(x), gamma = scipy_special_gamma(x),
    digamma = scipy_special_digamma(x), trigamma = scipy_special_polygamma(1, x),
    cumsum = jnp_cumsum(x), cumprod = jnp_cumprod(x),
    cummax = jnp_cumulative_max(x), cummin = jnp_cumulative_min(x),
    jax_error(.Generic, " is not supported for jax arrays"))
}

#' @rdname jax_operators
#' @export
Summary.jax_value <- function(..., na.rm = FALSE) {
  args <- list(...)
  one <- function(f) {
    vals <- lapply(args, f)
    if (length(vals) == 1L) vals[[1L]] else vals
  }
  combine <- function(vals, op) if (is.list(vals)) Reduce(op, vals) else vals
  switch(.Generic,
    sum = combine(one(jnp_sum), jnp_add),
    prod = combine(one(jnp_prod), jnp_multiply),
    max = combine(one(jnp_max), jnp_maximum),
    min = combine(one(jnp_min), jnp_minimum),
    all = combine(one(jnp_all), jnp_logical_and),
    any = combine(one(jnp_any), jnp_logical_or),
    range = jnp_stack(list(combine(one(jnp_min), jnp_minimum), combine(one(jnp_max), jnp_maximum))),
    jax_error(.Generic, " is not supported for jax arrays"))
}

#' @rdname jax_operators
#' @export
mean.jax_value <- function(x, ...) jnp_mean(x)

#' @rdname jax_operators
#' @export
t.jax_value <- function(x) jnp_transpose(x)

#' @export
all.equal.jax_value <- function(target, current, ...) {
  host <- function(v) {
    a <- as.array(v)
    if (length(dim(a)) <= 1L) as.vector(a) else a
  }
  all.equal(host(target), if (is_jax_value(current)) host(current) else current, ...)
}

strip_error <- function(what, instead) {
  force(what); force(instead)
  function(x, ...) jax_error(what, " would strip jax metadata or use R ordering; use ",
                             instead, " instead", class = "jax_type_error")
}

#' Base functions that are disabled for jax values
#'
#' `c()`, `rev()`, `rep()`, `aperm()`, `[[`, `dim<-` and `length<-` raise errors for
#' jax values because they would silently drop dtype information or use R's
#' column-major or 1-based conventions. Use `jnp_concatenate()`, `jnp_flip()`,
#' `jnp_tile()`, `jnp_transpose()`, `x[i]` (0-based) and `jnp_reshape()`, or convert
#' explicitly with `as.array()`.
#' @param x A jax value.
#' @param ... Ignored.
#' @param value Ignored.
#' @name jax_disabled_base
NULL
#' @rdname jax_disabled_base
#' @export
c.jax_value <- strip_error("c()", "jnp_concatenate() or jnp_stack()")
#' @rdname jax_disabled_base
#' @export
rev.jax_value <- strip_error("rev()", "jnp_flip()")
#' @rdname jax_disabled_base
#' @export
rep.jax_value <- strip_error("rep()", "jnp_tile() or jnp_repeat()")
#' @rdname jax_disabled_base
#' @param a,perm Ignored.
#' @export
aperm.jax_value <- function(a, perm, ...) strip_error("aperm()", "jnp_transpose() (0-based axes)")(a)
#' @rdname jax_disabled_base
#' @export
`[[.jax_value` <- strip_error("[[", "x[i] with 0-based indices")
#' @rdname jax_disabled_base
#' @export
`dim<-.jax_value` <- function(x, value) strip_error("dim<-", "jnp_reshape()")(x)
#' @rdname jax_disabled_base
#' @export
`length<-.jax_value` <- function(x, value) strip_error("length<-", "jnp_reshape() or jnp_pad()")(x)

#' @export
str.jax_value <- function(object, ...) cat(format(object), "\n")
