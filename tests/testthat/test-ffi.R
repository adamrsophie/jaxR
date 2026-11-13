# Port of the cases of tests/ffi_test.py that do not need jaxlib's own targets: a small
# handler is compiled against ffi_include_dir() and registered.

ffi_src <- '
#include <cstdint>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

static ffi::Error AxpbImpl(ffi::Buffer<ffi::F32> x, double scale, int64_t offset, bool negate,
                           ffi::Span<const int64_t> extra,
                           ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> total) {
  int64_t e = 0;
  for (auto v : extra) e += v;
  float s = 0;
  for (size_t i = 0; i < x.element_count(); ++i) {
    float v = static_cast<float>(x.typed_data()[i] * scale + offset + e);
    if (negate) v = -v;
    out->typed_data()[i] = v;
    s += v;
  }
  total->typed_data()[0] = s;
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(jaxr_test_axpb, AxpbImpl,
  ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Attr<double>("scale").Attr<int64_t>("offset")
    .Attr<bool>("negate").Attr<ffi::Span<const int64_t>>("extra")
    .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

// Copies the buffer memory as it is laid out.
static ffi::Error FlatImpl(ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> out) {
  for (size_t i = 0; i < x.element_count(); ++i) out->typed_data()[i] = x.typed_data()[i];
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(jaxr_test_flat, FlatImpl,
  ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

static ffi::Error FailImpl(ffi::Buffer<ffi::F32> x, ffi::ResultBuffer<ffi::F32> out) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, "jaxr test failure");
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(jaxr_test_fail, FailImpl,
  ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());
'

ffi_dll <- local({
  dll <- NULL
  function() {
    if (!is.null(dll)) return(dll)
    if (!nzchar(Sys.which("make"))) skip("no build tools")
    dir <- tempfile("jaxr_ffi")
    dir.create(dir)
    src <- file.path(dir, "handlers.cpp")
    writeLines(ffi_src, src)
    writeLines(c("CXX_STD = CXX17", paste0("PKG_CPPFLAGS = -I\"", ffi_include_dir(), "\"")),
               file.path(dir, "Makevars"))
    out <- suppressWarnings(system2(file.path(R.home("bin"), "R"),
                                    c("CMD", "SHLIB", shQuote(src)), stdout = TRUE, stderr = TRUE,
                                    env = paste0("R_MAKEVARS_USER=", file.path(dir, "Makevars"))))
    so <- file.path(dir, paste0("handlers", .Platform$dynlib.ext))
    if (!file.exists(so)) skip(paste("could not compile the FFI handlers:", paste(out, collapse = "\n")))
    dll <<- dyn.load(so)
    ffi_register_target("jaxr_test_axpb", getNativeSymbolInfo("jaxr_test_axpb", dll))
    ffi_register_target("jaxr_test_flat", list(execute = getNativeSymbolInfo("jaxr_test_flat", dll)))
    ffi_register_target("jaxr_test_fail", getNativeSymbolInfo("jaxr_test_fail", dll)$address)
    dll
  }
})

num <- function(x) as.numeric(as.array(x))

axpb <- function(vmap_method = NULL, n = 4L, has_side_effect = FALSE) {
  ffi_call("jaxr_test_axpb", list(ShapeDtypeStruct(n, "float32"), ShapeDtypeStruct(integer(0), "float32")),
           vmap_method = vmap_method, has_side_effect = has_side_effect)
}

test_that("ffi_call runs a registered handler eagerly and under jit", {
  skip_if_no_backend()
  ffi_dll()
  x <- jnp_arange(4, dtype = "float32")
  r <- axpb()(x, scale = 2, offset = 1L, negate = FALSE, extra = c(0L, 0L))
  expect_equal(num(r[[1]]), c(1, 3, 5, 7))
  expect_equal(num(r[[2]]), 16)
  r <- jit(function(x) axpb()(x, scale = 0.5, offset = 3L, negate = TRUE, extra = c(1L, 2L)))(x)
  expect_equal(num(r[[1]]), -(c(0, 0.5, 1, 1.5) + 6))
  # Attributes are part of the compiled call, not arguments.
  r <- axpb()(x, scale = 1, offset = 0L, negate = FALSE, extra = c(5L, 5L))
  expect_equal(num(r[[1]]), 0:3 + 10)
  # A single result.
  flat <- ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"))
  expect_equal(num(flat(x)), 0:3)
})

test_that("input layouts change the memory order the handler sees", {
  skip_if_no_backend()
  ffi_dll()
  m <- jnp_reshape(jnp_arange(6, dtype = "float32"), c(2L, 3L))
  row <- ffi_call("jaxr_test_flat", ShapeDtypeStruct(6L, "float32"))
  col <- ffi_call("jaxr_test_flat", ShapeDtypeStruct(6L, "float32"), input_layouts = list(c(1L, 0L)))
  expect_equal(num(row(m)), 0:5)
  expect_equal(num(col(m)), c(0, 3, 1, 4, 2, 5))
  expect_equal(num(jit(col)(m)), c(0, 3, 1, 4, 2, 5))
})

test_that("ffi_call under vmap follows vmap_method", {
  skip_if_no_backend()
  ffi_dll()
  xs <- jnp_reshape(jnp_arange(8, dtype = "float32"), c(2L, 4L))
  f <- function(m) function(x) axpb(m)(x, scale = 1, offset = 0L, negate = FALSE, extra = integer(0))[[2]]
  expect_equal(num(vmap(f("sequential"))(xs)), c(6, 22))
  expect_equal(num(vmap(f("sequential_unrolled"))(xs)), c(6, 22))
  expect_error(vmap(f(NULL))(xs), "vmap_method", class = "jax_not_implemented_error")
  # expand_dims: the handler sees the whole batch, so the total covers it.
  g <- function(x) ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"),
                            vmap_method = "expand_dims")(x)
  expect_equal(num(vmap(g)(xs)), num(xs))
})

test_that("ffi_call checks its arguments", {
  skip_if_no_backend()
  ffi_dll()
  x <- jnp_arange(4, dtype = "float32")
  expect_error(ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"), vmap_method = "bad"),
               "vmap_method must be on of", class = "jax_value_error")
  expect_error(ffi_call("t", ShapeDtypeStruct(4L, "float32"), legacy_backend_config = "x"),
               "legacy_backend_config", class = "jax_value_error")
  expect_error(ffi_call("t", list("a")), "must have 'shape' and 'dtype'", class = "jax_value_error")
  expect_error(ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"),
                        input_layouts = list(NULL, NULL))(x),
               "number of input arguments \\(1\\) must equal the number of input layouts \\(2\\)",
               class = "jax_value_error")
  expect_error(ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"),
                        input_output_aliases = c("1" = 0L))(x),
               "outside the range \\[0, 1\\)", class = "jax_value_error")
  expect_error(ffi_call("jaxr_test_flat", ShapeDtypeStruct(3L, "float32"),
                        input_output_aliases = c("0" = 0L))(x),
               "different abstract value", class = "jax_value_error")
  expect_equal(num(ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"),
                            input_output_aliases = c("0" = 0L))(x)), 0:3)
  expect_error(grad(function(x) jnp_sum(ffi_call("jaxr_test_flat", ShapeDtypeStruct(4L, "float32"))(x)))(x),
               "cannot be differentiated", class = "jax_value_error")
  expect_error(ffi_call("jaxr_test_fail", ShapeDtypeStruct(4L, "float32"))(x), "jaxr test failure")
  expect_error(ffi_register_target("t", 1), "external pointer", class = "jax_type_error")
  expect_error(ffi_register_target("t", list(prepare = 1)), "execute", class = "jax_not_implemented_error")
  expect_error(ffi_register_target("t", NULL, api_version = 0L), "api_version",
               class = "jax_not_implemented_error")
  expect_error(ffi_register_type_id("t", NULL), "not supported", class = "jax_value_error")
  expect_true(file.exists(file.path(ffi_include_dir(), "xla", "ffi", "api", "ffi.h")))
})

test_that("has_side_effect keeps an unused call in the jaxpr", {
  skip_if_no_backend()
  ffi_dll()
  x <- jnp_arange(4, dtype = "float32")
  f <- function(se) function(x) {
    axpb(has_side_effect = se)(x, scale = 1, offset = 0L, negate = FALSE, extra = integer(0))
    x + 1
  }
  expect_true(any(grepl("ffi_call", capture.output(print(make_jaxpr(f(TRUE))(x))))))
  expect_equal(num(jit(f(TRUE))(x)), 1:4)
  expect_equal(num(jit(f(FALSE))(x)), 1:4)
})
