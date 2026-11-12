# jax.numpy scalar types, dtype functions, constants, print options and array API metadata
# (ports of jax/_src/numpy/scalar_types.py, the dtype functions jax.numpy re-exports from
# jax/_src/dtypes.py and jax/_src/numpy/lax_numpy.py, and
# jax/_src/numpy/array_api_metadata.py). Dtypes are strings, as everywhere in jaxR.

# ---- dtype names ---------------------------------------------------------------------------------

# NumPy names of dtypes (np.dtype): unlike dtype(), 64-bit names are kept when x64 is
# disabled, and the generic names mean the 64-bit types.
misc_np_names <- c(dtype_aliases[setdiff(names(dtype_aliases), "float0")],
                   float = "float64", int = "int64", int_ = "int64", uint = "uint64",
                   complex = "complex128", csingle = "complex64", cdouble = "complex128")

misc_dtype <- function(x) {
  if (inherits(x, "jax_scalar_type")) return(attr(x, "dtype"))
  if (inherits(x, "jax_value")) return(aval_of(x)$dtype)
  if (is.character(x) && length(x) == 1L) {
    out <- misc_np_names[x]
    if (is.na(out)) jax_error("data type '", x, "' not understood", class = "jax_type_error")
    return(unname(out))
  }
  jax_error("Cannot interpret '", format(x), "' as a data type", class = "jax_type_error")
}

#' Data type objects
#'
#' Port of `jnp.dtype` (NumPy's `np.dtype`): the dtype named by a string, of a scalar
#' type such as [jnp_float32], or of an array. Unlike [dtype()], 64-bit names are not
#' canonicalized when x64 is disabled, and `"float"`, `"int"`, `"uint"` and `"complex"`
#' are the 64-bit types, as in NumPy.
#' @param obj A dtype name, scalar type or array.
#' @param align,copy Ignored (NumPy arguments).
#' @return A dtype string.
#' @export
jnp_dtype <- function(obj, align = FALSE, copy = FALSE) misc_dtype(obj)

# ---- scalar types ----------------------------------------------------------------------------------

# Port of scalar_types.py:_make_scalar_type: calling the type is jnp.asarray(x, dtype).
misc_scalar_type <- function(dt) {
  force(dt)
  structure(function(x) jnp_asarray(x, dtype = dt), dtype = dt,
            class = c("jax_scalar_type", "function"))
}

#' @export
print.jax_scalar_type <- function(x, ...) {
  cat("<jax.numpy scalar type ", attr(x, "dtype"), ">\n", sep = "")
  invisible(x)
}

#' Scalar types
#'
#' Ports of the `jax.numpy` scalar types. Each is a constructor:
#' `jnp_float32(x)` is `jnp_asarray(x, dtype = "float32")` (64-bit types give 32-bit
#' arrays unless x64 is enabled). They also work as dtype arguments of [jnp_dtype()],
#' [jnp_issubdtype()], [jnp_promote_types()], [jnp_can_cast()] and the info functions.
#' `jnp_int_`, `jnp_uint`, `jnp_float_` and `jnp_complex_` are the 64-bit types;
#' `jnp_single`, `jnp_double`, `jnp_csingle` and `jnp_cdouble` are aliases.
#'
#' The abstract types (`jnp_generic`, `jnp_number`, `jnp_integer`,
#' `jnp_signedinteger`, `jnp_unsignedinteger`, `jnp_inexact`, `jnp_floating`,
#' `jnp_complexfloating`, `jnp_character`, `jnp_flexible`) are category names for
#' [jnp_issubdtype()].
#' @param x Value(s) to convert.
#' @return An array of the type.
#' @name jnp_scalar_types
NULL

#' @rdname jnp_scalar_types
#' @export
jnp_bool_ <- misc_scalar_type("bool")
#' @rdname jnp_scalar_types
#' @export
jnp_bool <- jnp_bool_
#' @rdname jnp_scalar_types
#' @export
jnp_int8 <- misc_scalar_type("int8")
#' @rdname jnp_scalar_types
#' @export
jnp_int16 <- misc_scalar_type("int16")
#' @rdname jnp_scalar_types
#' @export
jnp_int32 <- misc_scalar_type("int32")
#' @rdname jnp_scalar_types
#' @export
jnp_int64 <- misc_scalar_type("int64")
#' @rdname jnp_scalar_types
#' @export
jnp_uint8 <- misc_scalar_type("uint8")
#' @rdname jnp_scalar_types
#' @export
jnp_uint16 <- misc_scalar_type("uint16")
#' @rdname jnp_scalar_types
#' @export
jnp_uint32 <- misc_scalar_type("uint32")
#' @rdname jnp_scalar_types
#' @export
jnp_uint64 <- misc_scalar_type("uint64")
#' @rdname jnp_scalar_types
#' @export
jnp_bfloat16 <- misc_scalar_type("bfloat16")
#' @rdname jnp_scalar_types
#' @export
jnp_float16 <- misc_scalar_type("float16")
#' @rdname jnp_scalar_types
#' @export
jnp_float32 <- misc_scalar_type("float32")
#' @rdname jnp_scalar_types
#' @export
jnp_float64 <- misc_scalar_type("float64")
#' @rdname jnp_scalar_types
#' @export
jnp_complex64 <- misc_scalar_type("complex64")
#' @rdname jnp_scalar_types
#' @export
jnp_complex128 <- misc_scalar_type("complex128")
#' @rdname jnp_scalar_types
#' @export
jnp_int1 <- misc_scalar_type("int1")
#' @rdname jnp_scalar_types
#' @export
jnp_uint1 <- misc_scalar_type("uint1")
#' @rdname jnp_scalar_types
#' @export
jnp_int2 <- misc_scalar_type("int2")
#' @rdname jnp_scalar_types
#' @export
jnp_uint2 <- misc_scalar_type("uint2")
#' @rdname jnp_scalar_types
#' @export
jnp_int4 <- misc_scalar_type("int4")
#' @rdname jnp_scalar_types
#' @export
jnp_uint4 <- misc_scalar_type("uint4")
#' @rdname jnp_scalar_types
#' @export
jnp_float4_e2m1fn <- misc_scalar_type("float4_e2m1fn")
#' @rdname jnp_scalar_types
#' @export
jnp_float6_e2m3fn <- misc_scalar_type("float6_e2m3fn")
#' @rdname jnp_scalar_types
#' @export
jnp_float6_e3m2fn <- misc_scalar_type("float6_e3m2fn")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e3m4 <- misc_scalar_type("float8_e3m4")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e4m3 <- misc_scalar_type("float8_e4m3")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e8m0fnu <- misc_scalar_type("float8_e8m0fnu")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e4m3b11fnuz <- misc_scalar_type("float8_e4m3b11fnuz")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e4m3fn <- misc_scalar_type("float8_e4m3fn")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e4m3fnuz <- misc_scalar_type("float8_e4m3fnuz")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e5m2 <- misc_scalar_type("float8_e5m2")
#' @rdname jnp_scalar_types
#' @export
jnp_float8_e5m2fnuz <- misc_scalar_type("float8_e5m2fnuz")
#' @rdname jnp_scalar_types
#' @export
jnp_single <- jnp_float32
#' @rdname jnp_scalar_types
#' @export
jnp_double <- jnp_float64
#' @rdname jnp_scalar_types
#' @export
jnp_csingle <- jnp_complex64
#' @rdname jnp_scalar_types
#' @export
jnp_cdouble <- jnp_complex128
#' @rdname jnp_scalar_types
#' @export
jnp_int_ <- jnp_int64
#' @rdname jnp_scalar_types
#' @export
jnp_uint <- jnp_uint64
#' @rdname jnp_scalar_types
#' @export
jnp_float_ <- jnp_float64
#' @rdname jnp_scalar_types
#' @export
jnp_complex_ <- jnp_complex128

# NumPy's abstract scalar type hierarchy (child = parent).
misc_category_parent <- c(number = "generic", integer = "number", signedinteger = "integer",
                          unsignedinteger = "integer", inexact = "number",
                          floating = "inexact", complexfloating = "inexact",
                          flexible = "generic", character = "flexible", generic = NA)

#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_generic <- "generic"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_number <- "number"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_integer <- "integer"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_signedinteger <- "signedinteger"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_unsignedinteger <- "unsignedinteger"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_inexact <- "inexact"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_floating <- "floating"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_complexfloating <- "complexfloating"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_character <- "character"
#' @rdname jnp_scalar_types
#' @format NULL
#' @export
jnp_flexible <- "flexible"

# ---- dtype functions --------------------------------------------------------------------------------

# Leaf category of a dtype in the NumPy hierarchy ("bool" sits directly under generic).
misc_category_of <- function(dt) {
  switch(dtype_kind(dt), b = "bool", i = "signedinteger", u = "unsignedinteger",
         f = "floating", c = "complexfloating")
}

misc_ancestors <- function(cat) {
  out <- character(0)
  while (!is.na(cat)) {
    out <- c(out, cat)
    cat <- if (cat == "bool") "generic" else misc_category_parent[[cat]]
  }
  out
}

misc_is_category <- function(x) {
  is.character(x) && length(x) == 1L && x %in% names(misc_category_parent)
}

#' Dtype queries and promotion
#'
#' Ports of `jnp.issubdtype`, `jnp.isdtype`, `jnp.promote_types`, `jnp.result_type`
#' and `jnp.can_cast`. Dtypes may be given as names, scalar types such as
#' [jnp_float32] or arrays. `promote_types` uses JAX's promotion lattice and does not
#' canonicalize 64-bit types; `result_type` promotes values (R scalars are weakly typed)
#' and canonicalizes like JAX. `can_cast` follows NumPy's casting rules.
#' @param arg1 A dtype, or an abstract category such as [jnp_floating].
#' @param arg2 A dtype or an abstract category.
#' @param dtype A dtype.
#' @param kind A dtype or one of `"bool"`, `"signed integer"`, `"unsigned integer"`,
#'   `"integral"`, `"real floating"`, `"complex floating"`, `"numeric"`, or a vector
#'   (list) of these.
#' @param a,b Dtypes.
#' @param ... Arrays, R values, dtypes or `NULL` (the default float type).
#' @param from_,to Dtypes (or arrays).
#' @param casting One of `"no"`, `"equiv"`, `"safe"`, `"same_kind"`, `"unsafe"`.
#' @return A logical scalar, or a dtype string.
#' @name jnp_dtype_functions
NULL

# Port of dtypes.py:_issubdtype_cached for the dtypes of jaxR.
#' @rdname jnp_dtype_functions
#' @export
jnp_issubdtype <- function(arg1, arg2) {
  a_cat <- if (misc_is_category(arg1)) arg1 else misc_category_of(misc_dtype(arg1))
  if (misc_is_category(arg2)) return(arg2 %in% misc_ancestors(a_cat))
  if (misc_is_category(arg1)) return(FALSE)
  misc_dtype(arg1) == misc_dtype(arg2)
}

misc_dtype_kinds <- list(
  bool = "bool",
  "signed integer" = c("int8", "int16", "int32", "int64"),
  "unsigned integer" = c("uint8", "uint16", "uint32", "uint64"),
  integral = c("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"),
  "real floating" = c("bfloat16", "float16", "float32", "float64"),
  "complex floating" = c("complex64", "complex128"),
  numeric = c("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
              "bfloat16", "float16", "float32", "float64", "complex64", "complex128"))

# Port of dtypes.py:isdtype.
#' @rdname jnp_dtype_functions
#' @export
jnp_isdtype <- function(dtype, kind) {
  the_dtype <- misc_dtype(dtype)
  kinds <- if (is.list(kind) && !inherits(kind, "jax_value")) kind
           else if (is.character(kind)) as.list(kind) else list(kind)
  options <- character(0)
  for (k in kinds) {
    if (is.character(k) && length(k) == 1L && k %in% names(misc_dtype_kinds)) {
      options <- c(options, misc_dtype_kinds[[k]])
      next
    }
    d <- tryCatch(misc_dtype(k), jax_error = function(e) NULL)
    if (is.null(d)) {
      if (is.character(k))
        jax_error("Unrecognized kind='", k, "' expected one of ['",
                  paste(names(misc_dtype_kinds), collapse = "', '"),
                  "'], or a compatible input for jnp.dtype()", class = "jax_value_error")
      jax_error("Expected kind to be a dtype, string, or tuple; got kind=", format(k),
                class = "jax_type_error")
    }
    options <- c(options, d)
  }
  the_dtype %in% options
}

# Port of dtypes.py:promote_types (the least upper bound on JAX's lattice; weak
# lattice points stand for the 64-bit types, as np.dtype(int) does).
#' @rdname jnp_dtype_functions
#' @export
jnp_promote_types <- function(a, b) {
  t <- least_upper_bound(c(misc_dtype(a), misc_dtype(b)))
  switch(t, "i*" = "int64", "f*" = "float64", "c*" = "complex128", t)
}

# Port of lax_numpy.py:result_type (dtypes.result_type: NULL is the default float).
#' @rdname jnp_dtype_functions
#' @export
jnp_result_type <- function(...) {
  args <- list(...)
  if (!length(args)) jax_error("at least one array or dtype is required", class = "jax_value_error")
  args <- lapply(args, function(a) {
    if (is.null(a)) default_float()
    else if (inherits(a, "jax_scalar_type")) attr(a, "dtype")
    else if (is.character(a)) misc_dtype(a)
    else a
  })
  do.call(result_type, args)
}

# NumPy casting tables (np.can_cast, including the ml_dtypes rules of bfloat16), as the
# dtypes each type casts to.
misc_safe_casts <- local({
  ints <- c("int8", "int16", "int32", "int64")
  uints <- c("uint8", "uint16", "uint32", "uint64")
  list(
    bool = c("bool", ints, uints, "bfloat16", "float16", "float32", "float64", "complex64", "complex128"),
    int8 = c(ints, "bfloat16", "float16", "float32", "float64", "complex64", "complex128"),
    int16 = c("int16", "int32", "int64", "float32", "float64", "complex64", "complex128"),
    int32 = c("int32", "int64", "float64", "complex128"),
    int64 = c("int64", "float64", "complex128"),
    uint8 = c("int16", "int32", "int64", uints, "bfloat16", "float16", "float32", "float64",
              "complex64", "complex128"),
    uint16 = c("int32", "int64", "uint16", "uint32", "uint64", "float32", "float64", "complex64",
               "complex128"),
    uint32 = c("int64", "uint32", "uint64", "float64", "complex128"),
    uint64 = c("uint64", "float64", "complex128"),
    bfloat16 = c("bfloat16", "float32", "float64", "complex64", "complex128"),
    float16 = c("float16", "float32", "float64", "complex64", "complex128"),
    float32 = c("float32", "float64", "complex64", "complex128"),
    float64 = c("float64", "complex128"),
    complex64 = c("complex64", "complex128"),
    complex128 = "complex128")
})

misc_same_kind_casts <- local({
  ints <- c("int8", "int16", "int32", "int64")
  uints <- c("uint8", "uint16", "uint32", "uint64")
  floats <- c("bfloat16", "float16", "float32", "float64")
  cplx <- c("complex64", "complex128")
  out <- list(bool = c("bool", ints, uints, floats, cplx))
  for (t in ints) out[[t]] <- c(ints, floats, cplx)
  for (t in uints) out[[t]] <- c(ints, uints, floats, cplx)
  out$bfloat16 <- c("bfloat16", "float32", "float64", cplx)
  for (t in c("float16", "float32", "float64")) out[[t]] <- c(floats, cplx)
  for (t in cplx) out[[t]] <- c("bfloat16", cplx)
  out
})

# NumPy's can_cast (dtypes.py:can_cast is np.can_cast).
#' @rdname jnp_dtype_functions
#' @export
jnp_can_cast <- function(from_, to, casting = "safe") {
  f <- misc_dtype(from_)
  t <- misc_dtype(to)
  switch(casting,
    no = , equiv = f == t,
    safe = t %in% misc_safe_casts[[f]],
    same_kind = t %in% misc_same_kind_casts[[f]],
    unsafe = TRUE,
    jax_error("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'",
              class = "jax_value_error"))
}

#' Machine limits of numeric types
#'
#' Ports of `jnp.finfo` (ml_dtypes.finfo) and `jnp.iinfo` (ml_dtypes.iinfo). The dtype
#' is not canonicalized, so `jnp_finfo("float64")` describes float64 even when x64 is
#' disabled. For complex types `finfo` describes the real component. 64-bit integer
#' bounds are rounded to the nearest double.
#' @param dtype A float or complex dtype, scalar type or array.
#' @param int_type An integer dtype, scalar type or array.
#' @return A list of limits: for `finfo`, `dtype`, `bits`, `eps`, `epsneg`, `max`,
#'   `min`, `tiny`, `smallest_normal`, `smallest_subnormal`, `resolution`, `precision`,
#'   `machep`, `negep`, `minexp`, `maxexp`, `nmant`, `nexp`, `iexp`; for `iinfo`,
#'   `dtype`, `kind`, `bits`, `min`, `max`.
#' @name jnp_finfo
NULL

#' @rdname jnp_finfo
#' @export
jnp_finfo <- function(dtype) {
  dt0 <- misc_dtype(dtype)
  dt <- complex_component(dt0)
  spec <- switch(dt, float16 = c(10, 5), bfloat16 = c(7, 8), float32 = c(23, 8),
                 float64 = c(52, 11), NULL)
  if (is.null(spec))
    jax_error("data type dtype('", dt0, "') not compatible with finfo", class = "jax_value_error")
  m <- spec[1L]
  e <- spec[2L]
  maxexp <- 2^(e - 1)
  minexp <- 2 - maxexp
  eps <- 2^-m
  precision <- floor(-log10(eps))
  # resolution is 10^-precision rounded to the type (ml_dtypes stores it as a scalar of it)
  res <- 10^-precision
  ulp <- 2^(floor(log2(res)) - m)
  mx <- (2 - eps) * 2^(maxexp - 1)
  list(dtype = dt, bits = dtype_bits(dt), eps = eps, epsneg = 2^-(m + 1), max = mx, min = -mx,
       tiny = 2^minexp, smallest_normal = 2^minexp, smallest_subnormal = 2^(minexp - m),
       resolution = if (m < 52) round(res / ulp) * ulp else res, precision = as.integer(precision), machep = as.integer(-m),
       negep = as.integer(-(m + 1)), minexp = as.integer(minexp), maxexp = as.integer(maxexp),
       nmant = as.integer(m), nexp = as.integer(e), iexp = as.integer(e))
}

#' @rdname jnp_finfo
#' @export
jnp_iinfo <- function(int_type) {
  dt <- misc_dtype(int_type)
  k <- dtype_kind(dt)
  if (!k %in% c("i", "u"))
    jax_error("Invalid integer data type '", k, "'.", class = "jax_value_error")
  b <- dtype_bits(dt)
  r <- if (k == "u") c(0, 2^b - 1) else c(-2^(b - 1), 2^(b - 1) - 1)
  list(dtype = dt, kind = k, bits = b, min = r[1L], max = r[2L])
}

# ---- constants ---------------------------------------------------------------------------------------

#' Constants
#'
#' The constants of `jax.numpy`: `jnp_pi`, `jnp_e`, `jnp_euler_gamma`, `jnp_inf`,
#' `jnp_nan` (R doubles, weakly typed when converted) and `jnp_newaxis` (the
#' [newaxis] index object).
#' @name jnp_constants
NULL

#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_pi <- pi
#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_e <- 0x1.5bf0a8b145769p+1
#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_euler_gamma <- 0x1.2788cfc6fb619p-1
#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_inf <- Inf
#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_nan <- NaN
#' @rdname jnp_constants
#' @format NULL
#' @export
jnp_newaxis <- newaxis

# ---- print options ---------------------------------------------------------------------------------

misc_print_defaults <- list(edgeitems = 3L, threshold = 1000L, floatmode = "maxprec",
                            precision = 8L, suppress = FALSE, linewidth = 75L, nanstr = "nan",
                            infstr = "inf", sign = "-", formatter = NULL, legacy = FALSE,
                            override_repr = NULL)

#' Print options
#'
#' Ports of `jnp.get_printoptions`, `jnp.set_printoptions` and `jnp.printoptions`
#' (NumPy's print options, which JAX uses to print arrays). The options are stored
#' per session with NumPy's defaults; `jnp_printoptions` sets them while `code` runs.
#' @param precision,threshold,edgeitems,linewidth,suppress,nanstr,infstr,formatter,sign,floatmode,legacy,override_repr
#'   NumPy print options; `NULL` leaves an option unchanged.
#' @param code Code to evaluate with the options set.
#' @param ... Options for `jnp_printoptions`, as for `jnp_set_printoptions`.
#' @return `jnp_get_printoptions` returns the list of options;
#'   `jnp_set_printoptions` returns the previous options invisibly; `jnp_printoptions`
#'   returns the value of `code`.
#' @name jnp_printoptions
NULL

#' @rdname jnp_printoptions
#' @export
jnp_get_printoptions <- function() .jx$printoptions %||% misc_print_defaults

#' @rdname jnp_printoptions
#' @export
jnp_set_printoptions <- function(precision = NULL, threshold = NULL, edgeitems = NULL,
                                 linewidth = NULL, suppress = NULL, nanstr = NULL,
                                 infstr = NULL, formatter = NULL, sign = NULL,
                                 floatmode = NULL, legacy = NULL, override_repr = NULL) {
  old <- jnp_get_printoptions()
  new <- list(precision = precision, threshold = threshold, edgeitems = edgeitems,
              linewidth = linewidth, suppress = suppress, nanstr = nanstr, infstr = infstr,
              formatter = formatter, sign = sign, floatmode = floatmode, legacy = legacy,
              override_repr = override_repr)
  .jx$printoptions <- update_params(old, Filter(Negate(is.null), new))
  invisible(old)
}

#' @rdname jnp_printoptions
#' @export
jnp_printoptions <- function(code, ...) {
  old <- .jx$printoptions
  jnp_set_printoptions(...)
  on.exit(.jx$printoptions <- old, add = TRUE)
  code
}

# ---- array API metadata ------------------------------------------------------------------------------

#' Array API metadata
#'
#' Ports of `jnp.__array_api_version__` and `jnp.__array_namespace_info__()` (from
#' jax/_src/numpy/array_api_metadata.py). The info object is a list of functions:
#' `capabilities()`, `default_device()`, `devices()`, `default_dtypes(device)` and
#' `dtypes(device, kind)`; `kind` is `NULL` (all numeric types and bool), a kind name
#' as in [jnp_isdtype()], or a vector of kind names.
#' @return `jnp_array_api_version` is a string; `jnp_array_namespace_info()` returns
#'   the info list.
#' @name jnp_array_api
NULL

#' @rdname jnp_array_api
#' @format NULL
#' @export
jnp_array_api_version <- "2025.12"

#' @rdname jnp_array_api
#' @export
jnp_array_namespace_info <- function() {
  api_types <- function() {
    t <- c("bool", "int8", "int16", "int32", "uint8", "uint16", "uint32", "float32", "complex64")
    if (x64_enabled()) t <- c(t, "int64", "uint64", "float64", "complex128")
    t
  }
  by_kind <- function(k) {
    if (!k %in% names(misc_dtype_kinds))
      jax_error("Unrecognized kind='", k, "'", class = "jax_value_error")
    t <- intersect(misc_dtype_kinds[[k]], api_types())
    stats::setNames(t, t)
  }
  list(
    capabilities = function() list("boolean indexing" = FALSE, "data-dependent shapes" = FALSE,
                                   "max dimensions" = 64L),
    default_device = function() NULL,
    devices = function() c(list(NULL), as.list(devices())),
    default_dtypes = function(device = NULL) {
      list("real floating" = default_float(), "complex floating" = default_complex(),
           integral = default_int(), indexing = default_int())
    },
    dtypes = function(device = NULL, kind = NULL) {
      if (is.null(kind)) return(c(by_kind("numeric"), by_kind("bool")))
      unlist(lapply(kind, by_kind))
    })
}
