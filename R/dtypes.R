# Dtypes and type promotion (port of jax/_src/dtypes.py).

.dtypes <- list(
  bool = list(kind = "b", bits = 8L, mlir = "i1"),
  int8 = list(kind = "i", bits = 8L, mlir = "i8"),
  int16 = list(kind = "i", bits = 16L, mlir = "i16"),
  int32 = list(kind = "i", bits = 32L, mlir = "i32"),
  int64 = list(kind = "i", bits = 64L, mlir = "i64"),
  uint8 = list(kind = "u", bits = 8L, mlir = "ui8"),
  uint16 = list(kind = "u", bits = 16L, mlir = "ui16"),
  uint32 = list(kind = "u", bits = 32L, mlir = "ui32"),
  uint64 = list(kind = "u", bits = 64L, mlir = "ui64"),
  float16 = list(kind = "f", bits = 16L, mlir = "f16"),
  bfloat16 = list(kind = "f", bits = 16L, mlir = "bf16"),
  float32 = list(kind = "f", bits = 32L, mlir = "f32"),
  float64 = list(kind = "f", bits = 64L, mlir = "f64"),
  complex64 = list(kind = "c", bits = 64L, mlir = "complex<f32>"),
  complex128 = list(kind = "c", bits = 128L, mlir = "complex<f64>"),
  float0 = list(kind = "V", bits = 0L, mlir = "i1"),
  # Narrow types of ml_dtypes (jax._src.dtypes._intn_dtypes and _custom_float_dtypes).
  int1 = list(kind = "i", bits = 1L, mlir = "i1"),
  uint1 = list(kind = "u", bits = 1L, mlir = "ui1"),
  int2 = list(kind = "i", bits = 2L, mlir = "i2"),
  uint2 = list(kind = "u", bits = 2L, mlir = "ui2"),
  int4 = list(kind = "i", bits = 4L, mlir = "i4"),
  uint4 = list(kind = "u", bits = 4L, mlir = "ui4"),
  float4_e2m1fn = list(kind = "f", bits = 4L, mlir = "f4E2M1FN"),
  float6_e2m3fn = list(kind = "f", bits = 6L, mlir = "f6E2M3FN"),
  float6_e3m2fn = list(kind = "f", bits = 6L, mlir = "f6E3M2FN"),
  float8_e3m4 = list(kind = "f", bits = 8L, mlir = "f8E3M4"),
  float8_e4m3 = list(kind = "f", bits = 8L, mlir = "f8E4M3"),
  float8_e8m0fnu = list(kind = "f", bits = 8L, mlir = "f8E8M0FNU"),
  float8_e4m3b11fnuz = list(kind = "f", bits = 8L, mlir = "f8E4M3B11FNUZ"),
  float8_e4m3fn = list(kind = "f", bits = 8L, mlir = "f8E4M3FN"),
  float8_e4m3fnuz = list(kind = "f", bits = 8L, mlir = "f8E4M3FNUZ"),
  float8_e5m2 = list(kind = "f", bits = 8L, mlir = "f8E5M2"),
  float8_e5m2fnuz = list(kind = "f", bits = 8L, mlir = "f8E5M2FNUZ")
)

ext_int_dtypes <- c("int1", "uint1", "int2", "uint2", "int4", "uint4")
ext_float_dtypes <- c("float4_e2m1fn", "float6_e2m3fn", "float6_e3m2fn", "float8_e3m4", "float8_e4m3",
                      "float8_e8m0fnu", "float8_e4m3b11fnuz", "float8_e4m3fn", "float8_e4m3fnuz",
                      "float8_e5m2", "float8_e5m2fnuz")
# Dtypes the host transfer code does not handle: they move through float64 / int32.
ext_dtypes <- c(ext_int_dtypes, ext_float_dtypes)
is_ext_dtype <- function(dt) dt %in% ext_dtypes

dtype_aliases <- c(
  bool = "bool", bool_ = "bool", logical = "bool",
  int8 = "int8", int16 = "int16", int32 = "int32", int64 = "int64",
  uint8 = "uint8", uint16 = "uint16", uint32 = "uint32", uint64 = "uint64",
  float16 = "float16", half = "float16", bfloat16 = "bfloat16", float32 = "float32",
  single = "float32", float64 = "float64", double = "float64",
  complex64 = "complex64", complex128 = "complex128", float0 = "float0",
  structure(c(ext_int_dtypes, ext_float_dtypes), names = c(ext_int_dtypes, ext_float_dtypes))
)

x64_enabled <- function() isTRUE(.jx$config$jax_enable_x64)

default_int <- function() if (x64_enabled()) "int64" else "int32"
default_uint <- function() if (x64_enabled()) "uint64" else "uint32"
default_float <- function() if (x64_enabled()) "float64" else "float32"
default_complex <- function() if (x64_enabled()) "complex128" else "complex64"

# Map a 64-bit dtype to its 32-bit counterpart when x64 is disabled.
canonicalize_dtype <- function(dt) {
  if (x64_enabled()) return(dt)
  switch(dt, int64 = "int32", uint64 = "uint32", float64 = "float32",
         complex128 = "complex64", dt)
}

#' Canonical dtype name
#'
#' Accepts dtype strings (`"float32"`, `"int32"`, `"bool"`, ...) and the generic
#' aliases `"float"`, `"int"`, `"uint"`, `"complex"` (which resolve to the default
#' types, 32-bit unless x64 is enabled), or a jax value (returns its dtype).
#' @param dtype A dtype name or a jax value.
#' @return The canonical dtype string.
#' @export
dtype <- function(dtype) {
  if (inherits(dtype, "jax_value")) return(aval_of(dtype)$dtype)
  if (inherits(dtype, "jax_prng_key")) return(paste0("key<", prng_key_impl(dtype)$tag, ">"))
  if (!is.character(dtype) || length(dtype) != 1L) jax_error("dtype must be a string")
  out <- switch(dtype,
    float = , float_ = , numeric = default_float(),
    int = , int_ = , integer = default_int(),
    uint = default_uint(),
    complex = , complex_ = default_complex(),
    dtype_aliases[dtype])
  if (is.na(out)) jax_error("unknown dtype '", dtype, "'")
  canonicalize_dtype(unname(out))
}

dtype_kind <- function(dt) .dtypes[[dt]]$kind
dtype_bits <- function(dt) .dtypes[[dt]]$bits
is_float_dtype <- function(dt) dtype_kind(dt) == "f"
is_int_dtype <- function(dt) dtype_kind(dt) %in% c("i", "u")
is_signed_dtype <- function(dt) dtype_kind(dt) == "i"
is_unsigned_dtype <- function(dt) dtype_kind(dt) == "u"
is_inexact_dtype <- function(dt) dtype_kind(dt) %in% c("f", "c")
is_complex_dtype <- function(dt) dtype_kind(dt) == "c"
is_bool_dtype <- function(dt) dt == "bool"

complex_component <- function(dt) switch(dt, complex64 = "float32", complex128 = "float64", dt)
complex_of <- function(dt) switch(dt, float64 = "complex128", complex128 = "complex128",
                                  "complex64")

#' Test dtype categories
#'
#' Port of `jnp.issubdtype` for the abstract categories `"floating"`, `"integer"`,
#' `"signedinteger"`, `"unsignedinteger"`, `"complexfloating"`, `"inexact"`,
#' `"number"`, `"bool"`.
#' @param arg1 A dtype or value.
#' @param arg2 A dtype or category name.
#' @return Logical scalar.
#' @export
issubdtype <- function(arg1, arg2) {
  dt <- if (inherits(arg1, "jax_value")) aval_of(arg1)$dtype else dtype(arg1)
  k <- dtype_kind(dt)
  switch(arg2,
    floating = k == "f", integer = k %in% c("i", "u"), signedinteger = k == "i",
    unsignedinteger = k == "u", complexfloating = k == "c", inexact = k %in% c("f", "c"),
    number = k %in% c("i", "u", "f", "c"), bool = , bool_ = k == "b", generic = TRUE,
    dt == dtype(arg2))
}

# Non-strict promotion lattice of jax._src.dtypes (int2/int4/float8 types omitted).
lattice_edges <- function() {
  list(
    bool = "i*", "i*" = c("uint8", "uint2", "uint4", "int8", "int2", "int4", "int1", "uint1"),
    uint8 = c("int16", "uint16"), uint16 = c("int32", "uint32"),
    uint32 = if (x64_enabled()) c("int64", "uint64") else c("int32", "uint64"),
    uint64 = "f*", int8 = "int16", int16 = "int32", int32 = "int64", int64 = "f*",
    "f*" = c(ext_float_dtypes, "bfloat16", "float16", "c*"), bfloat16 = "float32", float16 = "float32",
    float32 = c("float64", "complex64"), float64 = "complex128",
    "c*" = "complex64", complex64 = "complex128", complex128 = character()
  )
}

upper_bounds <- function(t, edges) {
  seen <- t
  frontier <- t
  while (length(frontier)) {
    nxt <- setdiff(unique(unlist(edges[frontier], use.names = FALSE)), seen)
    seen <- c(seen, nxt)
    frontier <- nxt
  }
  seen
}

least_upper_bound <- function(types) {
  types <- unique(types)
  if (length(types) == 1L) return(types)
  edges <- lattice_edges()
  ubs <- lapply(types, upper_bounds, edges = edges)
  common <- Reduce(intersect, ubs)
  for (t in common) if (setequal(upper_bounds(t, edges), common)) return(t)
  shown <- function(t) switch(t, "i*" = "int", "f*" = "float", "c*" = "complex", t)
  tup <- paste0("(", paste0("'", vapply(types, shown, ""), "'", collapse = ", "), ")")
  why <- if (any(grepl("^float8", types))) "8-bit floats" else if (any(grepl("^float6", types))) "6-bit floats"
         else if (any(grepl("^float4", types))) "4-bit floats" else if (any(types %in% ext_int_dtypes)) "intn"
         else ""
  msg <- if (why == "intn") {
    paste0("Input dtypes ", tup, " have no available implicit dtype promotion path. To avoid unintended ",
           "promotion, 1-bit, 2-bit and 4-bit integers do not support implicit promotion. If you'd like ",
           "your inputs to be promoted to another type, you can do so explicitly using e.g. x.astype('int32')")
  } else if (nzchar(why)) {
    paste0("Input dtypes ", tup, " have no available implicit dtype promotion path. To avoid unintended ",
           "promotion, ", why, " do not support implicit promotion. If you'd like your inputs to be ",
           "promoted to another type, you can do so explicitly using e.g. x.astype('float32')")
  } else {
    paste0("Input dtypes ", tup, " have no available implicit dtype promotion path. Try explicitly ",
           "casting inputs to the desired output type.")
  }
  jax_error(msg, class = c("jax_type_promotion_error", "jax_type_error"))
}

lattice_type <- function(dt, weak) {
  if (!weak || dt == "bool") return(dt)
  switch(dtype_kind(dt), i = , u = "i*", f = "f*", c = "c*")
}

from_lattice <- function(t) {
  switch(t, "i*" = default_int(), "f*" = default_float(), "c*" = default_complex(),
         canonicalize_dtype(t))
}

# Result dtype and weak flag (jax._src.dtypes._lattice_result_type).
promote_types <- function(dts, weaks) {
  if (length(dts) == 1L) return(list(dtype = dts, weak = weaks))
  if (identical(.jx$config$jax_numpy_dtype_promotion, "strict")) {
    strong <- unique(dts[!weaks])
    if (length(strong) > 1L)
      jax_error("Input dtypes (", paste0("'", strong, "'", collapse = ", "), ") have no available ",
                "implicit dtype promotion path when jax_numpy_dtype_promotion=strict. Try ",
                "explicitly casting inputs to the desired output type, or set ",
                "jax_numpy_dtype_promotion=standard.", class = "jax_type_promotion_error")
  }
  if (any(dts == "float0")) jax_error("float0 arrays do not support arithmetic",
                                      class = "jax_type_error")
  if (length(unique(dts)) == 1L && !all(weaks)) return(list(dtype = dts[1L], weak = FALSE))
  if (all(weaks)) {
    t <- least_upper_bound(dts)
    weak <- TRUE
  } else {
    t <- least_upper_bound(mapply(lattice_type, dts, weaks, USE.NAMES = FALSE))
    weak <- t %in% c("i*", "f*", "c*")
  }
  dt <- from_lattice(t)
  list(dtype = dt, weak = weak && dt != "bool")
}

#' Result dtype of values under JAX type promotion
#' @param ... Arrays, tracers, R values or dtype strings.
#' @return A dtype string.
#' @export
result_type <- function(...) {
  avs <- lapply(list(...), function(a) {
    if (is.character(a)) list(dtype = dtype(a), weak = FALSE) else aval_of(a)
  })
  promote_types(vapply(avs, `[[`, "", "dtype"), vapply(avs, `[[`, NA, "weak"))$dtype
}

#' @rdname result_type
#' @param type1,type2 Dtypes.
#' @export
promote_types_dtype <- function(type1, type2) {
  # Explicit names are not canonicalized (jnp.promote_types("float64", "float32") is
  # float64 without x64), and a weak lattice result is the 64 bit type, as np.dtype of
  # the Python scalar type is.
  ts <- c(dtype_info_name(type1), dtype_info_name(type2))
  if (identical(.jx$config$jax_numpy_dtype_promotion, "strict"))
    return(promote_types(ts, c(FALSE, FALSE))$dtype)
  t <- least_upper_bound(ts)
  switch(t, "i*" = "int64", "f*" = "float64", "c*" = "complex128", t)
}

# Like dtype() but explicit names are not canonicalized (jnp.finfo(float64) describes
# float64 even when x64 is disabled); generic aliases still resolve to the defaults.
dtype_info_name <- function(dtype) {
  if (is.character(dtype) && length(dtype) == 1L && !is.na(dtype_aliases[dtype]))
    return(unname(dtype_aliases[dtype]))
  dtype(dtype)
}

#' Information about a floating point dtype
#' @param dtype A float or complex dtype.
#' @return A list with `bits`, `eps`, `max`, `min`, `tiny`, `smallest_normal`, `resolution`.
#' @export
finfo <- function(dtype = "float") {
  dt <- complex_component(dtype_info_name(dtype))
  # m: mantissa bits, e: exponent bits; max and tiny of the formats of ml_dtypes that
  # reserve no infinity (fn, fnuz, fnu) or use a different bias are given explicitly.
  spec <- switch(dt,
    float16 = list(m = 10, e = 5),
    bfloat16 = list(m = 7, e = 8),
    float32 = list(m = 23, e = 8),
    float64 = list(m = 52, e = 11),
    float4_e2m1fn = list(m = 1, e = 2, max = 6, tiny = 1),
    float6_e2m3fn = list(m = 3, e = 2, max = 7.5, tiny = 1),
    float6_e3m2fn = list(m = 2, e = 3, max = 28, tiny = 0.25),
    float8_e3m4 = list(m = 4, e = 3, max = 15.5, tiny = 0.25),
    float8_e4m3 = list(m = 3, e = 4, max = 240, tiny = 2^-6),
    float8_e8m0fnu = list(m = 0, e = 8, max = 2^127, tiny = 2^-127),
    float8_e4m3b11fnuz = list(m = 3, e = 4, max = 30, tiny = 2^-10),
    float8_e4m3fn = list(m = 3, e = 4, max = 448, tiny = 2^-6),
    float8_e4m3fnuz = list(m = 3, e = 4, max = 240, tiny = 2^-7),
    float8_e5m2 = list(m = 2, e = 5, max = 57344, tiny = 2^-14),
    float8_e5m2fnuz = list(m = 2, e = 5, max = 57344, tiny = 2^-15),
    jax_error("finfo requires a floating dtype, got ", dt))
  emax <- 2^(spec$e - 1) - 1
  mx <- spec$max %||% ((2 - 2^-spec$m) * 2^emax)
  tiny <- spec$tiny %||% 2^(1 - emax)
  mn <- if (dt == "float8_e8m0fnu") tiny else -mx
  # ml_dtypes uses precision 1 for the 8-bit floats and 0 for the 4 and 6 bit ones, with
  # the resolution rounded to the dtype.
  res <- switch(dt,
    float4_e2m1fn = , float6_e2m3fn = , float6_e3m2fn = 1,
    float8_e3m4 = 0.09375, float8_e5m2 = , float8_e5m2fnuz = 0.09375, float8_e8m0fnu = 0.125,
    float8_e4m3 = , float8_e4m3fn = , float8_e4m3fnuz = , float8_e4m3b11fnuz = 0.1015625,
    {
      # 10^-precision rounded to the dtype, as np.finfo and ml_dtypes report it.
      r <- 10^-floor(spec$m * log10(2))
      q <- 2^(floor(log2(r)) - spec$m)
      round(r / q) * q
    })
  list(dtype = dt, bits = dtype_bits(dt), eps = 2^-spec$m, max = mx, min = mn,
       tiny = tiny, smallest_normal = tiny, resolution = res, nmant = spec$m, nexp = spec$e)
}

#' Information about an integer dtype
#' @param dtype An integer dtype.
#' @return A list with `bits`, `min`, `max`.
#' @export
iinfo <- function(dtype = "int") {
  dt <- dtype_info_name(dtype)
  if (!is_int_dtype(dt)) jax_error("iinfo requires an integer dtype, got ", dt)
  b <- dtype_bits(dt)
  r <- if (is_unsigned_dtype(dt)) c(0, 2^b - 1) else c(-2^(b - 1), 2^(b - 1) - 1)
  list(dtype = dt, bits = b, min = r[1L], max = r[2L])
}

# Natural dtype of R data.
dtype_of_r <- function(x) {
  if (is.logical(x)) "bool"
  else if (is.integer(x)) "int32"
  else if (is.double(x)) canonicalize_dtype("float64")
  else if (is.complex(x)) canonicalize_dtype("complex128")
  else jax_error("cannot convert R object of type '", typeof(x), "' to a jax array")
}

mlir_elem_type <- function(dt) .dtypes[[dt]]$mlir
