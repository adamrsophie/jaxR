# Dtype system: canonicalization, the pairwise promotion table, finfo/iinfo and
# issubdtype, all compared with the wheel (fixtures/dtypes_x32.json, dtypes_x64.json).

fx_dtype_operand <- function(name) {
  switch(name, weak_int = 1L, weak_float = 1, weak_complex = 1i, py_bool = TRUE, name)
}

fx_array_operand <- function(name) {
  switch(name, weak_int = 1L, weak_float = 1, weak_complex = 1i, py_bool = TRUE,
         jnp_zeros(integer(0), name))
}

for (x64 in c(FALSE, TRUE)) local({
  x64 <- x64
  tag <- if (x64) "x64" else "x32"

  test_that(sprintf("pairwise result_type table matches the wheel [%s]", tag), {
    skip_if_no_backend()
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      bad <- character()
      for (e in fx$result_type) {
        a <- fx_dtype_operand(e$a)
        b <- fx_dtype_operand(e$b)
        got <- tryCatch(result_type(a, b), error = function(err) paste("error:", conditionMessage(err)))
        avs <- lapply(list(a, b), function(v) {
          if (is.character(v)) list(dtype = dtype(v), weak = FALSE) else jaxR:::aval_of(v)
        })
        weak <- jaxR:::promote_types(vapply(avs, `[[`, "", "dtype"), vapply(avs, `[[`, NA, "weak"))$weak
        if (!identical(got, e$dtype) || !identical(weak, e$weak_type))
          bad <- c(bad, sprintf("%s + %s: got %s%s, wheel %s%s", e$a, e$b, got, if (weak) " (weak)" else "",
                                e$dtype, if (e$weak_type) " (weak)" else ""))
      }
      expect_identical(bad, character())
    })
  })

  test_that(sprintf("jnp_add promotes like jnp.add for every dtype pair [%s]", tag), {
    skip_if_no_backend()
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      bad <- character()
      for (e in fx$result_type) {
        r <- tryCatch(jnp_add(fx_array_operand(e$a), fx_array_operand(e$b)),
                      error = function(err) NULL)
        got <- if (is.null(r)) "error" else jaxR:::aval_of(r)$dtype
        weak <- !is.null(r) && jaxR:::aval_of(r)$weak
        if (!identical(got, e$add_dtype) || !identical(weak, e$add_weak_type))
          bad <- c(bad, sprintf("%s + %s: got %s%s, wheel %s%s", e$a, e$b, got, if (weak) " (weak)" else "",
                                e$add_dtype, if (e$add_weak_type) " (weak)" else ""))
      }
      expect_identical(bad, character())
    })
  })

  test_that(sprintf("result_type of several arguments matches the wheel [%s]", tag), {
    skip_if_no_backend()
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      for (e in fx$result_type_multi) {
        args <- lapply(unlist(e$args), fx_dtype_operand)
        expect_identical(do.call(result_type, args), e$dtype, info = e$code)
      }
    })
  })

  test_that(sprintf("dtype() canonicalizes like dtypes.canonicalize_dtype [%s]", tag), {
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      for (e in fx$canonicalize) expect_identical(dtype(e$r), e$dtype, info = e$code)
    })
  })

  test_that(sprintf("finfo matches jnp.finfo [%s]", tag), {
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      for (e in fx$finfo) {
        fi <- finfo(e$dtype)
        expect_identical(fi$dtype, e$result_dtype, info = e$code)
        expect_identical(as.integer(fi$bits), as.integer(e$bits), info = e$code)
        expect_identical(as.integer(fi$nmant), as.integer(e$nmant), info = e$code)
        expect_identical(as.integer(fi$nexp), as.integer(e$nexp), info = e$code)
        for (k in c("eps", "max", "min", "tiny", "smallest_normal"))
          expect_identical(fi[[k]], fx_num1(e[[k]]), info = paste(e$code, k))
        # jnp.finfo stores resolution rounded to the dtype (float16(0.001) = 0.0010004).
        expect_equal(fi$resolution, fx_num1(e$resolution), tolerance = fi$eps, info = e$code)
      }
    })
  })

  test_that(sprintf("iinfo matches jnp.iinfo [%s]", tag), {
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      for (e in fx$iinfo) {
        ii <- iinfo(e$dtype)
        expect_identical(ii$dtype, e$result_dtype, info = e$code)
        expect_identical(as.integer(ii$bits), as.integer(e$bits), info = e$code)
        expect_identical(ii$min, fx_num1(e$min), info = e$code)
        expect_identical(ii$max, fx_num1(e$max), info = e$code)
      }
    })
  })

  test_that(sprintf("issubdtype matches jnp.issubdtype [%s]", tag), {
    fx <- fx_load("dtypes", x64)
    with_jax_config("jax_enable_x64", x64, {
      bad <- character()
      for (e in fx$issubdtype) {
        if (!identical(issubdtype(e$dtype, e$category), e$value))
          bad <- c(bad, e$code)
      }
      expect_identical(bad, character())
    })
  })
})

test_that("iinfo and finfo of 64-bit types do not depend on x64", {
  with_jax_config("jax_enable_x64", FALSE, {
    expect_identical(iinfo("int64")$bits, 64L)
    expect_identical(finfo("float64")$bits, 64L)
  })
})

test_that("R integer scalars are weak default ints (int64 under x64), like Python ints", {
  skip_if_no_backend()
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(dtype(as_jax(1L)), "int64")
    expect_identical(dtype(jnp_add(1L, 2L)), "int64")
  })
})

test_that("dtype() accepts jax values and rejects unknown names", {
  skip_if_no_backend()
  expect_identical(dtype(as_jax(1:3)), "int32")
  expect_identical(dtype(as_jax(TRUE)), "bool")
  expect_error(dtype("float128"), class = "jax_error")
  expect_error(dtype(c("float32", "int32")), "dtype must be a string")
  expect_error(dtype(3), "dtype must be a string")
})

test_that("promote_types_dtype is jnp.promote_types without weak types", {
  with_jax_config("jax_enable_x64", TRUE, {
    expect_identical(promote_types_dtype("int8", "uint8"), "int16")
    expect_identical(promote_types_dtype("bfloat16", "float16"), "float32")
    expect_identical(promote_types_dtype("uint64", "int8"), "float64")
    expect_identical(promote_types_dtype("int32", "complex64"), "complex64")
    expect_identical(promote_types_dtype("int64", "float32"), "float32")
  })
  with_jax_config("jax_enable_x64", FALSE, {
    expect_identical(promote_types_dtype("int8", "uint8"), "int16")
    expect_identical(promote_types_dtype("float32", "int32"), "float32")
  })
})

test_that("float0 does not take part in arithmetic promotion", {
  expect_error(jaxR:::promote_types(c("float0", "float32"), c(FALSE, FALSE)),
               class = "jax_type_error")
})
