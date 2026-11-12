# Interoperability with symcompR (symbolic computation, a SymPy port by the same author).

sym_fn <- function(name) getExportedValue("symcompR", name)

# A symbolic number (Integer, Rational, Float or a numeric constant such as pi) becomes
# a weakly typed jax scalar; expressions with free symbols are rejected.
sym_to_jax_constant <- function(x, dtype = NULL) {
  v <- tryCatch(as.double(x), error = function(e) NULL, warning = function(w) NULL)
  if (is.null(v) || length(v) != 1L || is.na(v))
    jax_error("symcompR expressions with free symbols cannot be mixed with jax arrays; ",
              "convert them to functions first with jax_lambdify()", class = "jax_type_error")
  as_jax(v, dtype)
}

# SymPy function names and their jaxR counterparts (looked up at call time).
.sym_function_map <- c(
  sin = "jnp_sin", cos = "jnp_cos", tan = "jnp_tan", asin = "jnp_arcsin",
  acos = "jnp_arccos", atan = "jnp_arctan", sinh = "jnp_sinh", cosh = "jnp_cosh",
  tanh = "jnp_tanh", asinh = "jnp_arcsinh", acosh = "jnp_arccosh", atanh = "jnp_arctanh",
  exp = "jnp_exp", log = "jnp_log", sqrt = "jnp_sqrt", cbrt = "jnp_cbrt", Abs = "jnp_abs",
  sign = "jnp_sign", floor = "jnp_floor", ceiling = "jnp_ceil", atan2 = "jnp_arctan2",
  re = "jnp_real", im = "jnp_imag", conjugate = "jnp_conjugate", Heaviside = "jnp_heaviside",
  gamma = "scipy_special_gamma", loggamma = "scipy_special_gammaln",
  digamma = "scipy_special_digamma", polygamma = "scipy_special_polygamma",
  erf = "scipy_special_erf", erfc = "scipy_special_erfc", erfinv = "scipy_special_erfinv",
  beta = "scipy_special_beta", zeta = "scipy_special_zeta", factorial = "scipy_special_factorial",
  expit = "scipy_special_expit", logit = "scipy_special_logit", sinc = "jnp_sinc",
  besseli = "scipy_special_iv", besselj = "scipy_special_jv", LambertW = "scipy_special_lambertw",
  uppergamma = "jaxr_uppergamma", lowergamma = "jaxr_lowergamma"
)

jaxr_lowergamma <- function(a, x) jnp_multiply(scipy_special_gammainc(a, x), scipy_special_gamma(a))
jaxr_uppergamma <- function(a, x) jnp_multiply(scipy_special_gammaincc(a, x), scipy_special_gamma(a))

sym_eval <- function(e, vals) {
  st <- sym_fn("sym_type")
  sa <- sym_fn("sym_args")
  sn <- sym_fn("sym_name")
  ev <- function(e) {
    t <- st(e)
    switch(t,
      Integer = , Rational = , Float = as_jax(as.double(e)),
      Symbol = , Dummy = {
        nm <- sn(e)
        if (is.null(vals[[nm]])) jax_error("jax_lambdify: no value for symbol '", nm, "'")
        vals[[nm]]
      },
      Constant = , NumberSymbol = , ImaginaryUnit = , Infinity = , NegativeInfinity = , "NaN" = {
        nm <- sn(e) %||% t
        switch(nm, I = , ImaginaryUnit = as_jax(1i), oo = , Infinity = as_jax(Inf),
               `-oo` = , NegativeInfinity = as_jax(-Inf), nan = , "NaN" = as_jax(NaN),
               as_jax(as.double(e)))
      },
      BooleanAtom = , BooleanTrue = , BooleanFalse = as_jax(identical(sn(e), "true") || t == "BooleanTrue"),
      Add = Reduce(jnp_add, lapply(sa(e), ev)),
      Mul = Reduce(jnp_multiply, lapply(sa(e), ev)),
      Pow = {
        a <- sa(e)
        et <- st(a[[2L]])
        if (et %in% c("Integer", "Rational", "Float")) {
          p <- as.double(a[[2L]])
          if (p == 0.5) return(jnp_sqrt(ev(a[[1L]])))
          return(jnp_power(ev(a[[1L]]), p))
        }
        jnp_power(ev(a[[1L]]), ev(a[[2L]]))
      },
      Function = {
        nm <- sn(e)
        args <- lapply(sa(e), ev)
        if (nm %in% c("Max", "Min"))
          return(Reduce(if (nm == "Max") jnp_maximum else jnp_minimum, args))
        if (nm == "log" && length(args) == 2L)
          return(jnp_true_divide(jnp_log(args[[1L]]), jnp_log(args[[2L]])))
        target <- .sym_function_map[nm]
        if (is.na(target) || !exists(target, envir = asNamespace("jaxR"), inherits = FALSE))
          jax_error("jax_lambdify: no jaxR translation for symcompR function '", nm, "'")
        do.call(get(target, envir = asNamespace("jaxR")), args)
      },
      Piecewise = {
        out <- NULL
        for (pair in rev(sa(e))) {
          p <- sa(pair)
          val <- ev(p[[1L]])
          cond <- ev(p[[2L]])
          out <- if (is.null(out)) jnp_where(cond, val, jnp_full_like(val, NaN)) else jnp_where(cond, val, out)
        }
        out
      },
      Equality = do.call(jnp_equal, lapply(sa(e), ev)),
      Unequality = do.call(jnp_not_equal, lapply(sa(e), ev)),
      StrictLessThan = do.call(jnp_less, lapply(sa(e), ev)),
      LessThan = do.call(jnp_less_equal, lapply(sa(e), ev)),
      StrictGreaterThan = do.call(jnp_greater, lapply(sa(e), ev)),
      GreaterThan = do.call(jnp_greater_equal, lapply(sa(e), ev)),
      And = Reduce(jnp_logical_and, lapply(sa(e), ev)),
      Or = Reduce(jnp_logical_or, lapply(sa(e), ev)),
      Xor = Reduce(jnp_logical_xor, lapply(sa(e), ev)),
      Not = jnp_logical_not(ev(sa(e)[[1L]])),
      jax_error("jax_lambdify: cannot translate symcompR expressions of type '", t, "'"))
  }
  ev(e)
}

#' Turn a symcompR expression into a jaxR function
#'
#' The analogue of SymPy's `lambdify(args, expr, modules = "jax")`: the returned function
#' evaluates a symcompR expression with jaxR operations, so it can be differentiated with
#' [grad()], vectorized with [vmap()] and compiled with [jit()].
#' @param args A symcompR symbol or a list of symbols (the function's arguments, in order).
#' @param expr A symcompR expression, or a list of expressions (the function then returns a
#'   list).
#' @return An R function of jax arrays.
#' @export
jax_lambdify <- function(args, expr) {
  if (!requireNamespace("symcompR", quietly = TRUE))
    jax_error("jax_lambdify() needs the symcompR package")
  syms <- if (inherits(args, "sym")) list(args) else args
  nms <- vapply(syms, function(s) sym_fn("sym_name")(s), "")
  multi <- is.list(expr) && !inherits(expr, "sym")
  f <- function(...) {
    vals <- list(...)
    if (length(vals) != length(nms))
      jax_error("jax_lambdify function expects ", length(nms), " arguments, got ", length(vals))
    vals <- lapply(vals, as_jax)
    names(vals) <- nms
    if (multi) lapply(expr, sym_eval, vals = vals) else sym_eval(expr, vals)
  }
  f
}
