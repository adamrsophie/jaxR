# Invariants of constants, control flow bounds, pytree registrations, AOT stages,
# callback lifetimes, sparse batching, ODE budgets and NumPy files.

vals <- function(x) as.vector(as.array(x))

x64 <- function(code) with_jax_config("jax_enable_x64", TRUE, code)

test_that("signed zeros survive constant splats", {
  skip_if_no_backend()
  x64({
    w <- as_jax(c(0, -0))
    f <- custom_jvp(function(x) w, function(p, t) list(w, jnp_zeros_like(w)))
    expect_identical(1 / vals(f(as_jax(1))), c(Inf, -Inf))
    expect_identical(1 / vals(jit(f)(as_jax(1))), c(Inf, -Inf))
  })
})

test_that("tree matching tells registered subclasses apart", {
  reg <- jaxR:::.pytree_registry
  on.exit(rm(list = intersect(c("inv_base", "inv_sub"), ls(reg)), envir = reg), add = TRUE)
  make <- function(a, b, cls) structure(list(a = a, b = b), class = cls)
  register_pytree_node("inv_base", function(x) list(children = list(x$a, x$b), aux = NULL),
                       function(aux, ch) make(ch[[1L]], ch[[2L]], "inv_base"))
  register_pytree_node("inv_sub", function(x) list(children = list(x$b, x$a), aux = NULL),
                       function(aux, ch) make(ch[[2L]], ch[[1L]], c("inv_sub", "inv_base")))
  a <- make(1, 2, "inv_base")
  b <- make(10, 20, c("inv_sub", "inv_base"))
  expect_error(tree_map(`+`, a, b), class = "jax_value_error")
  expect_error(tree_map(`+`, b, a), class = "jax_value_error")
  # An unregistered subclass uses the handler of its registered class.
  r <- tree_map(`+`, a, make(10, 20, c("inv_plain", "inv_base")))
  expect_identical(c(r$a, r$b), c(11, 22))
})

test_that("registering a class again does not reuse traces of the old registration", {
  skip_if_no_backend()
  reg <- jaxR:::.pytree_registry
  on.exit(rm(list = "inv_rereg", envir = reg), add = TRUE)
  make <- function(a, b) structure(list(a = a, b = b), class = "inv_rereg")
  register_pytree_node("inv_rereg", function(p) list(children = list(p$a, p$b), aux = NULL),
                       function(aux, ch) make(ch[[1L]], ch[[2L]]))
  f <- jit(function(p) p$a + 10 * p$b)
  p <- make(as_jax(1), as_jax(2))
  expect_equal(vals(f(p)), 21)
  register_pytree_node("inv_rereg", function(p) list(children = list(p$b, p$a), aux = NULL),
                       function(aux, ch) make(ch[[2L]], ch[[1L]]))
  expect_equal(vals(f(p)), 21)
})

test_that("concrete fori bounds count trips exactly", {
  skip_if_no_backend()
  x64({
    lo <- as_jax(-2147483647, "int32"); hi <- as_jax(2147483647, "int32")
    f <- function(c) lax_fori_loop(lo, hi, function(i, c) c + 1, c)
    expect_error(f(as_jax(0)), class = "jax_value_error")
    expect_error(jit(f)(as_jax(0)), class = "jax_value_error")
    expect_equal(vals(lax_fori_loop(hi, lo, function(i, c) c + 1, as_jax(0))), 0)
    n <- lax_fori_loop(as_jax(-100L, "int8"), as_jax(100L, "int8"), function(i, c) c + 1, as_jax(0))
    expect_equal(vals(n), 200)
  })
})

test_that("scan rejects invalid lengths without stacked outputs", {
  skip_if_no_backend()
  body <- function(c, x) list(c + 1, NULL)
  expect_error(lax_scan(body, as_jax(0), length = -2L), class = "jax_value_error")
  expect_error(lax_scan(body, as_jax(0), length = 1.5), class = "jax_value_error")
  expect_equal(vals(lax_scan(body, as_jax(0), length = 0L)[[1L]]), 0)
  expect_equal(vals(lax_scan(body, as_jax(0), length = 3L)[[1L]]), 3)
  expect_equal(vals(lax_scan(body, as_jax(0), length = as_jax(3L))[[1L]]), 3)
})

test_that("switch clamps a wide index before narrowing", {
  skip_if_no_backend()
  x64({
    f <- function(i) lax_switch(i, list(function() 10, function() 20, function() 30))
    for (case in list(c(2^32, 30), c(-2^32 + 1, 10))) {
      i <- as_jax(case[[1L]], "int64")
      expect_equal(vals(f(i)), case[[2L]])
      expect_equal(vals(jit(f)(i)), case[[2L]])
    }
  })
  many <- lapply(0:299, function(k) { force(k); function() k })
  expect_equal(vals(lax_switch(as_jax(250L, "uint8"), many)), 250)
  expect_equal(vals(lax_switch(as_jax(-5L, "int8"), many)), 0)
})

test_that("all static AOT stages take no arguments", {
  skip_if_no_backend()
  f <- jit(function(n) jnp_arange(n), static_argnames = "n")
  expect_equal(vals(f$lower(3L)$compile()()), c(0, 1, 2))
  expect_equal(vals(f$lower(n = 3L)$compile()()), c(0, 1, 2))
})

test_that("AOT stages reject outside Refs and leave them unchanged", {
  skip_if_no_backend()
  ref <- new_ref(as_jax(0))
  f <- jit(function(x) { ref[] <- ref[] + x; ref[] })
  expect_error(f$trace(as_jax(2)), class = "jax_not_implemented_error")
  expect_error(f$lower(as_jax(2)), class = "jax_not_implemented_error")
  expect_error(f$eval_shape(as_jax(2)), class = "jax_not_implemented_error")
  expect_equal(vals(ref[]), 0)
  expect_equal(vals(f(as_jax(2))), 2)
  expect_equal(vals(ref[]), 2)
  # make_jaxpr() and eval_shape() trace without changing the Ref.
  make_jaxpr(function(x) { ref[] <- ref[] + x; ref[] })(as_jax(1))
  eval_shape(function(x) { ref[] <- ref[] + x; ref[] }, as_jax(1))
  expect_equal(vals(ref[]), 2)
  # A Ref made inside the staged function is fine.
  g <- jit(function(x) { r <- new_ref(x); r[] <- r[] * 2; r[] })
  expect_equal(vals(g$lower(as_jax(3))$compile()(as_jax(3))), 6)
})

test_that("half precision callback results round once", {
  skip_if_no_backend()
  for (case in list(list("float16", 1 + 2^-11 + 2^-25, 1 + 2^-10),
                    list("bfloat16", 1 + 2^-8 + 2^-30, 1 + 2^-7))) {
    v <- case[[2L]]
    f <- jit(function(x) pure_callback(function(y) v, ShapeDtypeStruct(integer(0), case[[1L]]), x))
    expect_equal(vals(f(as_jax(0, "float32"))), case[[3L]], info = case[[1L]])
  }
})

test_that("a retained Lowered stage compiles after clear_caches", {
  skip_if_no_backend()
  cb <- function(v) v + 1
  x <- as_jax(4, "float32")
  f <- jit(function(v) pure_callback(cb, ShapeDtypeStruct(integer(0), "float32"), v))
  expect_equal(vals(f(x)), 5)
  lowered <- f$lower(x)
  clear_caches()
  expect_equal(vals(lowered$compile()(x)), 5)
})

test_that("captures of discarded AOT programs are released", {
  skip_if_no_backend()
  probe <- function(compile) {
    env <- new.env()
    env$big <- 1
    cb <- local(function(v) v + big, env)
    f <- jit(function(v) pure_callback(cb, ShapeDtypeStruct(integer(0), "float32"), v))
    lowered <- f$lower(as_jax(1, "float32"))
    if (compile) {
      compiled <- lowered$compile()
      compiled(as_jax(1, "float32"))
    }
    .Call(jaxR:::jaxr_weakref_new, env)
  }
  for (compile in c(FALSE, TRUE)) {
    w <- probe(compile)
    for (i in 1:3) { clear_caches(); gc() }
    expect_null(.Call(jaxR:::jaxr_weakref_get, w), label = paste("compile =", compile))
  }
})

test_that("sparse data gradients compose with vmap", {
  skip_if_no_backend()
  x64({
    for (fmt in c("BCOO", "BCSR")) {
      cl <- get(paste0("experimental_sparse_", fmt))
      m <- cl$fromdense(diag(c(2, 3)), nse = 3L)
      build <- if (fmt == "BCOO") function(d) cl(list(d, m$indices), shape = c(2L, 2L))
               else function(d) cl(list(d, m$indices, m$indptr), shape = c(2L, 2L))
      f <- grad(function(d, v) jnp_sum(build(d) %*% v))
      V <- as_jax(rbind(c(5, 7), c(10, 14)))
      expect_equal(as.array(vmap(f, in_axes = list(NULL, 0L))(m$data, V)),
                   rbind(c(5, 7, 0), c(10, 14, 0)), info = fmt)
    }
  })
})

test_that("odeint signals an exhausted step budget", {
  skip_if_no_backend()
  x64({
    out <- vals(experimental_ode_odeint(function(y, t) -y, 1, as_jax(c(0, 10)), mxstep = 1L))
    expect_equal(out[1L], 1)
    expect_true(is.nan(out[2L]))
  })
})

test_that("NPY payloads are checked and NPZ writes are safe", {
  skip_if_no_backend()
  d <- tempfile("npio")
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  path <- file.path(d, "full.npy")
  jnp_save(path, as_jax(c(1, 2, 3)))
  bytes <- readBin(path, "raw", n = 1e5)
  hl <- as.integer(bytes[[9]]) + 256L * as.integer(bytes[[10]])
  short <- file.path(d, "short.npy")
  writeBin(bytes[seq_len(10L + hl + 4L)], short)
  expect_error(jnp_load(short), class = "jax_value_error")
  args <- c(list(file = file.path(d, "out.npz")), setNames(list(as_jax(1)), "../escape"))
  expect_error(do.call(jnp_savez, args), class = "jax_value_error")
  expect_false(file.exists(file.path(dirname(d), "escape.npy")))
  # A failed write keeps the previous file.
  target <- file.path(d, "prev.npz")
  jnp_savez(target, a = as_jax(1))
  prior <- readBin(target, "raw", n = 1e5)
  old <- Sys.getenv("R_ZIPCMD", unset = NA_character_)
  on.exit(if (is.na(old)) Sys.unsetenv("R_ZIPCMD") else Sys.setenv(R_ZIPCMD = old), add = TRUE)
  Sys.setenv(R_ZIPCMD = file.path(d, "no_zip_program"))
  expect_error(suppressWarnings(jnp_savez(target, a = as_jax(2))))
  expect_identical(readBin(target, "raw", n = 1e5), prior)
  expect_length(setdiff(list.files(d, all.files = TRUE, no.. = TRUE),
                        c("full.npy", "short.npy", "prev.npz")), 0L)
})

test_that("a jit function under vmap retraces for a new named axis size", {
  skip_if_no_backend()
  m <- vmap(jit(function(x) lax_pmean(x, "i")), axis_name = "i")
  expect_equal(vals(m(as_jax(c(1, 3)))), c(2, 2))
  expect_equal(vals(m(as_jax(c(1, 3, 5)))), c(3, 3, 3))
  # An unnamed inner vmap forwards collectives of the outer axis.
  r <- vmap(vmap(function(x) lax_psum(x, "i")), axis_name = "i")(
    as_jax(matrix(c(1, 2, 3, 4), 2, byrow = TRUE)))
  expect_equal(as.array(r), matrix(c(4, 6, 4, 6), 2, byrow = TRUE))
})

test_that("complex construction and where handle broadcasting and imaginary parts", {
  skip_if_no_backend()
  im <- as_jax(c(0, 0, 0), "float32")
  expect_equal(vals(grad(function(x) jnp_sum(lax_real(lax_complex(x, im))))(as_jax(2, "float32"))), 3)
  expect_equal(vals(jnp_where(as_jax(c(0+1i, 0+0i)), 1L, 2L)), c(1L, 2L))
})

test_that("abstract tracing restores the whole state of outside Refs", {
  skip_if_no_backend()
  r <- new_ref(as_jax(1))
  f <- jit(function(x) { r[] <- x; x })
  eval_shape(f, as_jax(2))
  expect_equal(vals(r[] + 1), 2)
  q <- new_ref(as_jax(1L))
  expect_error(jit(function(x) { free_ref(q); x })$lower(as_jax(0L)), class = "jax_not_implemented_error")
  expect_equal(vals(q[]), 1L)
  eval_shape(function(x) { freeze(q); x }, as_jax(0L))
  q[] <- as_jax(5L)
  expect_equal(vals(q[]), 5L)
})

test_that("treedefs keep the rules of their registration", {
  reg <- jaxR:::.pytree_registry
  on.exit(rm(list = "inv_keep", envir = reg), add = TRUE)
  mk <- function(a, b) structure(list(a = a, b = b), class = "inv_keep")
  register_pytree_node("inv_keep", function(p) list(children = list(p$a, p$b), aux = NULL),
                       function(aux, ch) mk(ch[[1L]], ch[[2L]]))
  fl <- tree_flatten(mk(1, 2))
  register_pytree_node("inv_keep", function(p) list(children = list(p$b, p$a), aux = NULL),
                       function(aux, ch) mk(ch[[2L]], ch[[1L]]))
  u <- tree_unflatten(fl$treedef, fl$leaves)
  expect_identical(c(u$a, u$b), c(1, 2))
})

test_that("compiled stages accept dicts with their names in another order", {
  skip_if_no_backend()
  g <- jit(function(p) p$a - p$b)$lower(list(a = as_jax(1L), b = as_jax(2L)))$compile()
  expect_equal(vals(g(list(b = as_jax(2L), a = as_jax(1L)))), -1L)
})

test_that("transposed CSR products ignore unused capacity", {
  skip_if_no_backend()
  m <- experimental_sparse_CSR(list(as_jax(c(2, 0), "float32"), as_jax(c(0L, 1L)), as_jax(c(0L, 1L, 1L))),
                               shape = c(2L, 2L))
  expect_equal(vals(m$T %*% as_jax(c(3, Inf), "float32")), c(6, 0))
})

test_that("dot products honor and encode precision", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2), "float32")
  t <- default_matmul_precision("highest", lower_text(function(v) jnp_dot(v, v), x))
  expect_match(t, "precision HIGHEST>, #stablehlo<precision HIGHEST>", fixed = TRUE)
  t <- lower_text(function(v) jnp_dot(v, v, precision = c("high", "highest")), x)
  expect_match(t, "[#stablehlo<precision HIGH>, #stablehlo<precision HIGHEST>]", fixed = TRUE)
})

test_that("the norm logpdf JVP stays finite for large tangents", {
  skip_if_no_backend()
  f32 <- function(v) as_jax(v, "float32")
  r <- jvp(function(x, l, s) scipy_stats_norm_logpdf(x, l, s), list(f32(2), f32(0), f32(2)),
           list(f32(2e38), f32(-2e38), f32(0)))
  expect_equal(vals(r[[2L]]), -2e38, tolerance = 1e-6)
})

test_that("npy headers parse with whitespace around their colons", {
  skip_if_no_backend()
  hdr <- "{'descr': '<i2', 'fortran_order' : True, 'shape': (2, 2), }"
  pad <- (64 - (11 + nchar(hdr)) %% 64) %% 64
  h <- paste0(hdr, strrep(" ", pad), "\n")
  p <- tempfile(fileext = ".npy")
  on.exit(unlink(p), add = TRUE)
  con <- file(p, "wb")
  writeBin(as.raw(c(0x93, charToRaw("NUMPY"), 1, 0)), con)
  writeBin(as.integer(nchar(h)), con, size = 2, endian = "little")
  writeBin(charToRaw(h), con)
  writeBin(1:4, con, size = 2, endian = "little")
  close(con)
  expect_equal(as.array(jnp_load(p)), matrix(c(1L, 2L, 3L, 4L), 2))
})

test_that("diagonals are strided slices", {
  skip_if_no_backend()
  x <- jnp_reshape(jnp_arange(24L), c(2L, 3L, 4L))
  for (off in c(-2L, -1L, 0L, 1L, 3L)) {
    expect_equal(as.array(jnp_linalg_diagonal(x, offset = off)),
                 as.array(jnp_diagonal(x, offset = off, axis1 = -2L, axis2 = -1L)), info = off)
  }
  m <- matrix(1:12, 3)
  expect_equal(vals(jnp_linalg_diagonal(m)), diag(m))
  expect_equal(vals(jnp_linalg_diagonal(m, offset = 1L)), diag(m[, -1]))
})
