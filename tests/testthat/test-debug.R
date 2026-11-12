# jax.debug: callback, print, log and breakpoint, eager, under jit, vmap and grad.

test_that("debug_DebugEffect names", {
  expect_identical(debug_DebugEffect, "Debug")
  expect_identical(debug_OrderedDebugEffect, "OrderedDebug")
})

test_that("debug_print formats like str.format with R formatting of values", {
  skip_if_no_backend()
  expect_output(debug_print("x = {x}", x = as_jax(1.5)), "^x = 1.5$")
  expect_output(debug_print("{} and {}", as_jax(1L), as_jax(TRUE)), "^1 and TRUE$")
  expect_output(debug_print("{0}-{0}-{1}", as_jax(2), as_jax(3)), "^2-2-3$")
  expect_output(debug_print("{x:.3f} {y:5d}|", x = as_jax(pi), y = as_jax(42L)), "^3.142    42\\|$")
  expect_output(debug_print("{x} {{y}}", x = as_jax(1)), "^1 \\{y\\}$")
  expect_output(debug_print("{s}: {x}", s = "label", x = as_jax(2)), "^label: 2$")
  expect_output(debug_print("v = {v}", v = as_jax(c(1, 2, 3))), "v = [1] 1 2 3", fixed = TRUE)
  expect_output(debug_print("m =\n{m}", m = as_jax(matrix(1:4, 2))), "[1,]    1    3", fixed = TRUE)
  expect_output(debug_print("{x}", x = as_jax(0.1)), "^0.1$")
  expect_output(debug_print("hello", as_jax(1), as_jax(2), skip_format_check = TRUE), "^hello 1 2$")
  expect_output(debug_print("plain text"), "^plain text$")
  expect_output(debug_print(ordered = TRUE)("two-call {x}", x = as_jax(5L)), "^two-call 5$")
  expect_output(debug_print("{a} and {b}", a = 1, b = list(1, "z")), 'list(1, z)', fixed = TRUE)
})

test_that("debug_print checks its format string like jax.debug.print", {
  skip_if_no_backend()
  expect_error(debug_print("hello", as_jax(1)), "Unused positional arguments to `jax.debug.print`",
               fixed = TRUE)
  expect_error(debug_print("{x}", x = 1, y = 2),
               "Unused keyword arguments to `jax.debug.print`: ['y']. You may be passing an f-string",
               fixed = TRUE)
  expect_error(debug_print("{y}", x = 1), "KeyError: 'y'")
  expect_error(debug_print("{} {}", 1), "Replacement index 1 out of range")
  expect_error(debug_print("{} {0}", 1), "cannot switch from automatic field numbering")
  expect_error(debug_print("{x", x = 1), "expected '}' before end of string")
  expect_error(debug_print("x}", x = 1), "Single '}' encountered")
  expect_error(debug_print("{x.a}", x = 1), "plain field names only")
  expect_error(debug_print("{x:>5}", x = as_jax(1)), "unsupported format spec")
  expect_error(debug_print(NULL, 1), "debug_print received unexpected arguments in the two-call form")
  expect_error(debug_print("plain", x = 1, skip_format_check = TRUE),
               "Format without placeholders should not have kwargs.")
})

test_that("debug_print runs inside jit, once per element under vmap, and under grad", {
  skip_if_no_backend()
  f <- jit(function(x) {
    debug_print("x = {x}", x = x)
    x * 2
  })
  expect_output(r <- f(as_jax(3)), "^x = 3$")
  expect_jax_equal(r, 6)
  expect_output(f(as_jax(4)), "^x = 4$")
  expect_output(vmap(function(x) { debug_print("e {}", x); x })(as_jax(c(7, 8))), "e 7\ne 8")
  expect_output(jit(vmap(function(x) { debug_print("e {}", x); x }))(as_jax(c(7, 8))), "e 7\ne 8")
  expect_output(g <- grad(function(x) { debug_print("g {x}", x = x); x^2 })(as_jax(3)), "^g 3$")
  expect_jax_equal(g, 6)
  expect_output(g2 <- jit(grad(function(x) { debug_print("g {x}", x = x); x^2 }))(as_jax(3)), "^g 3$")
  expect_jax_equal(g2, 6)
  expect_output(jit(function(x) { debug_print("first"); debug_print("second {}", x, ordered = TRUE); x })(
    as_jax(1)), "first\nsecond 1")
  jp <- make_jaxpr(function(x) { debug_print("x = {x}", x = x); x })(1)
  expect_identical(jp$jaxpr$eqns[[1L]]$prim$name, "debug_print")
  expect_message(debug_log("logged {}", as_jax(1)), "logged 1")
  # inside compiled code R runs the callback at top level: capture the message stream
  msg <- utils::capture.output(invisible(jit(function(x) { debug_log("logged {}", x); x })(as_jax(2))),
                               type = "message")
  expect_identical(msg, "logged 2")
})

test_that("debug_callback passes R values, static arguments, and is transparent", {
  skip_if_no_backend()
  got <- list()
  cb <- function(x, tag, ...) got[[length(got) + 1L]] <<- list(x = x, tag = tag, dots = list(...))
  debug_callback(cb, as_jax(c(1, 2)), "eager", extra = 5L)
  expect_identical(got[[1L]]$x, array(c(1, 2), 2L))
  expect_identical(got[[1L]]$tag, "eager")
  expect_identical(got[[1L]]$dots, list(extra = 5))
  debug_callback(ordered = TRUE)(cb, as_jax(3), "two-call")
  expect_identical(got[[2L]]$tag, "two-call")
  expect_error(debug_callback(1, 2), "first argument to jax.debug.callback must be callable")
  expect_error(debug_callback(ordered = TRUE, 3), "must be callable")
  expect_null(debug_callback(cb, 1, "x"))
  got <- list()
  f <- jit(function(x) {
    debug_callback(cb, x, "jit")
    x + 1
  })
  expect_jax_equal(f(as_jax(c(1, 2))), c(2, 3))
  expect_identical(got[[1L]]$x, array(c(1, 2), 2L))
  expect_identical(got[[1L]]$tag, "jit")
  got <- list()
  vmap(function(x) { debug_callback(cb, x, "v"); x })(as_jax(matrix(1:4, 2)))
  expect_identical(lapply(got, `[[`, "x"), list(array(c(1, 3), 2L), array(c(2, 4), 2L)))
  got <- list()
  expect_jax_equal(grad(function(x) { debug_callback(cb, x, "g"); jnp_sum(x^2) })(as_jax(c(1, 2))), c(2, 4))
  expect_identical(got[[1L]]$x, array(c(1, 2), 2L))
  # inside control flow
  got <- list()
  lax_fori_loop(0L, 3L, function(i, c) { debug_callback(cb, c, "loop"); c + 1 }, as_jax(0))
  expect_identical(vapply(got, function(g) as.numeric(g$x), 1), c(0, 1, 2))
})

test_that("debug_breakpoint captures the frame variables", {
  skip_if_no_backend()
  frames <- NULL
  keep <- function(fr) frames <<- fr
  h <- function(x) {
    y <- x * 2
    label <- "here"
    debug_breakpoint(backend = keep, num_frames = 2L)
    y
  }
  h(as_jax(3))
  expect_length(frames, 2L)
  expect_identical(frames[[1L]]$label, "here")
  expect_jax_equal(frames[[1L]]$y, 6)
  expect_null(debug_breakpoint(backend = keep, num_frames = 1L))
  expect_error(debug_breakpoint(ordered = TRUE, token = 1), "mutually exclusive")
  tok <- debug_breakpoint(backend = keep, token = as_jax(c(1, 2)))
  expect_jax_equal(tok, c(1, 2))
  frames <- NULL
  h1 <- function(x) {
    y <- x * 2
    label <- "here"
    debug_breakpoint(backend = keep, num_frames = 1L)
    y
  }
  jit(h1)(as_jax(5))
  expect_equal(as.numeric(frames[[1L]]$y), 10)
  expect_identical(frames[[1L]]$label, "here")
})
