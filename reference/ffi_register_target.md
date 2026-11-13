# Foreign function interface

Ports of `jax.ffi.register_ffi_target`, `jax.ffi.register_ffi_type`,
`jax.ffi.register_ffi_type_id`,
`jax.ffi.register_ffi_target_as_batch_partitionable` and
`jax.ffi.include_dir`.

## Usage

``` r
ffi_register_target(name, fn, platform = "cpu", api_version = 1L, ...)

ffi_register_type(name, type_registration, platform = "cpu")

ffi_register_type_id(name, obj, platform = "cpu")

ffi_register_target_as_batch_partitionable(name)

ffi_include_dir()
```

## Arguments

- name:

  The name of the target (or of the type).

- fn:

  The handler: the address of the symbol defined with
  `XLA_FFI_DEFINE_HANDLER_SYMBOL`, as an external pointer or the result
  of
  [`getNativeSymbolInfo()`](https://rdrr.io/r/base/getNativeSymbolInfo.html);
  or a named list with element `execute` holding it (the other FFI
  stages, `instantiate`, `prepare` and `initialize`, cannot be
  registered through the PJRT plugin interface jaxR uses).

- platform:

  The target platform: `"cpu"` (the default), `"cuda"`, `"rocm"`, or the
  platform name XLA uses.

- api_version:

  The custom call API of the handler: 1 (the default) for the typed FFI.
  Handlers of the earlier custom call API (0) cannot be registered
  through the PJRT plugin interface.

- ...:

  Not used (JAX passes extra arguments to
  `xla_client.register_custom_call_target`).

- type_registration:

  For `ffi_register_type()`: a list with element `type_id`, the address
  of the `XLA_FFI_TypeId` of the type (filled in by XLA when it holds
  0), and optionally `type_info`, the address of its `XLA_FFI_TypeInfo`.

- obj:

  Not used.

## Value

`ffi_register_target()`, `ffi_register_type()` and
`ffi_register_target_as_batch_partitionable()` return `NULL` invisibly;
`ffi_include_dir()` returns the directory of the XLA FFI headers.

## Details

A foreign function is a typed XLA FFI handler written in C or C++ with
the headers in `ffi_include_dir()` (`#include "xla/ffi/api/ffi.h"`, then
`XLA_FFI_DEFINE_HANDLER_SYMBOL`), compiled into a shared library and
loaded with [`dyn.load()`](https://rdrr.io/r/base/dynload.html).
`ffi_register_target()` registers it under a name;
[`ffi_call()`](https://adamrsophie.github.io/jaxR/reference/ffi_call.md)
calls it.

## Examples

``` r
if (FALSE) { # \dontrun{
dir <- tempfile("ffi")
dir.create(dir)
writeLines(c(
  '#include "xla/ffi/api/ffi.h"',
  'namespace ffi = xla::ffi;',
  'static ffi::Error AddOne(ffi::Buffer<ffi::F32> x, float y,',
  '                         ffi::ResultBuffer<ffi::F32> out) {',
  '  for (size_t i = 0; i < x.element_count(); ++i)',
  '    out->typed_data()[i] = x.typed_data()[i] + y;',
  '  return ffi::Error::Success();',
  '}',
  'XLA_FFI_DEFINE_HANDLER_SYMBOL(add_one, AddOne,',
  '  ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::F32>>().Attr<float>("y")',
  '    .Ret<ffi::Buffer<ffi::F32>>());'), file.path(dir, "add_one.cc"))
writeLines(c("CXX_STD = CXX17", paste0("PKG_CPPFLAGS = -I", ffi_include_dir())),
           file.path(dir, "Makevars"))
system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", file.path(dir, "add_one.cc")),
        env = paste0("R_MAKEVARS_USER=", file.path(dir, "Makevars")))
dll <- dyn.load(file.path(dir, paste0("add_one", .Platform$dynlib.ext)))
ffi_register_target("add_one", getNativeSymbolInfo("add_one", dll))
x <- jnp_arange(4, dtype = "float32")
ffi_call("add_one", ShapeDtypeStruct(4L, "float32"))(x, y = as_jax(1, "float32"))
} # }
```
