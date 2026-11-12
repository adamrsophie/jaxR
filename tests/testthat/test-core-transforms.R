# Transformations (grad, value_and_grad, jvp, vjp, linearize, vmap, jit, eval_shape,
# linear_transpose, ravel_pytree) and their compositions, against the wheel
# (fixtures/transforms_*.json). The functions mirror the Python functions the fixtures
# were generated from.

R_FUNS <- list(
  sin_sq = function(x) jnp_sum(jnp_sin(x)^2),
  poly = function(x) jnp_sum(3 * x^3 - 2 * x + 1),
  softplus = function(x) jnp_sum(jnp_log1p(jnp_exp(x))),
  tanh_mv = function(W, x) jnp_sum(jnp_tanh(jnp_matmul(W, x))),
  max_ties = function(x) jnp_max(x),
  maximum = function(x, y) jnp_sum(jnp_maximum(x, y)),
  abs_sum = function(x) jnp_sum(jnp_abs(x)),
  div = function(x, y) jnp_sum(x / y),
  pow = function(x, y) jnp_sum(x^y),
  where = function(x) jnp_sum(jnp_where(x > 0, x^2, -x)),
  reshape_t = function(x) {
    jnp_sum(jnp_transpose(jnp_reshape(x, c(3L, 2L))) * jnp_reshape(jnp_arange(6, dtype = "float"), c(2L, 3L)))
  },
  concat_slice = function(x) jnp_sum(lax_slice(jnp_concatenate(list(x, 2 * x)), 1L, 5L)^2),
  pad = function(x) jnp_sum(lax_pad(x, 0, list(c(1L, 2L, 1L))) * jnp_arange(10, dtype = "float")),
  mean_var = function(x) jnp_var(x) + jnp_mean(x),
  logistic = function(x) jnp_sum(lax_logistic(x) * x),
  atan2 = function(x, y) jnp_sum(jnp_arctan2(x, y)),
  erf_sqrt = function(x) jnp_sum(lax_erf(x) * jnp_sqrt(x)),
  batch_mm = function(a, b) jnp_sum(jnp_matmul(a, b)^2),
  min_reduce = function(x) {
    jnp_sum(jnp_min(x, axis = 0L) * lax_index_in_dim(jnp_max(x, axis = 1L), 0L, keepdims = FALSE))
  },
  dyn_slice = function(x) {
    jnp_sum(lax_dynamic_slice(x, 1L, 2L)^2) + jnp_sum(lax_dynamic_update_slice(x, jnp_ones(2L), 2L) * x)
  },
  rev_bcast = function(x) jnp_sum(lax_rev(x, 0L) * lax_broadcast_in_dim(x, c(2L, 3L), 1L)),
  select_clamp = function(x) jnp_sum(lax_clamp(-1, x, 1) * lax_select(x > 0, x, 2 * x)),
  dict_fn = function(p) jnp_sum(p$w * lax_index_in_dim(p$b, 0L, keepdims = FALSE)) + jnp_sum(p$b^2),
  vec_sin = function(x) jnp_sin(x) * x,
  two_out = function(x, y) list(x * y, jnp_sin(x) + y),
  aux = function(x) list(jnp_sum(x^2), x * 2),
  scalar_cube = function(x) x^3 + jnp_sin(x),
  dot = function(x, y) jnp_dot(x, y),
  affine = function(x) jnp_sum(2 * x),
  lin2 = function(x, y) list(x + 2 * y, jnp_sum(x)),
  int_cast = function(x) jnp_sum(jnp_astype(x, "int32") * x),
  exp_log = function(x) jnp_sum(jnp_exp(x) * jnp_log(x)),
  cumul = function(x) jnp_sum(x * jnp_sum(x))
)
RF <- R_FUNS

# Mirrors of the Python helpers of the fixtures (custom derivative rules and branches).
cj_sin <- custom_jvp(function(x) jnp_sin(x), jvp = function(primals, tangents) {
  list(jnp_sin(primals[[1L]]), 3 * jnp_cos(primals[[1L]]) * tangents[[1L]])
})
cj_pow <- custom_jvp(function(n, x) x^n, nondiff_argnums = 0L, jvp = function(n, primals, tangents) {
  list(primals[[1L]]^n, 10 * n * primals[[1L]]^(n - 1) * tangents[[1L]])
})
cj_exp <- custom_jvp_defjvps(custom_jvp(function(x) jnp_exp(x)), function(t, ans, x) 2 * ans * t)
cv_sq <- custom_vjp(function(x) x^2, fwd = function(x) list(x^2, x), bwd = function(res, ct) list(3 * res * ct))
cv_two <- custom_vjp(function(x, y) x * y, fwd = function(x, y) list(x * y, list(x, y)),
                     bwd = function(res, ct) list(res[[2L]] * ct * 2, NULL))
SW <- list(function(x) x + 1, function(x) x * 2, function(x) -x)
JF <- function(x) jnp_sin(x) * jnp_sum(x^2)
i32 <- function(v) as_jax(v, "int32")

# R counterparts of the fixture expressions, keyed by case id.
R_CASES <- list(
  "grad_has_aux/aux" = function(a0) grad(RF$aux, has_aux = TRUE)(a0),
  "value_and_grad/sin_sq" = function(a0) value_and_grad(RF$sin_sq)(a0),
  "value_and_grad_argnums1/tanh_mv" = function(a0, a1) value_and_grad(RF$tanh_mv, argnums = 1L)(a0, a1),
  "value_and_grad_has_aux/aux" = function(a0) value_and_grad(RF$aux, has_aux = TRUE)(a0),
  "jvp/vec_sin" = function(a0, a1) jvp(RF$vec_sin, list(a0), list(a1)),
  "jvp/two_out" = function(a0, a1, a2) jvp(RF$two_out, list(a0, a1), list(a2, a2)),
  "jvp/scalar_cube" = function(a0, a1) jvp(RF$scalar_cube, list(a0), list(a1)),
  "vjp/vec_sin" = function(a0, a1) vjp(RF$vec_sin, a0)[[2L]](a1),
  "vjp_out/vec_sin" = function(a0) vjp(RF$vec_sin, a0)[[1L]],
  "vjp/two_out" = function(a0, a1, a2) vjp(RF$two_out, a0, a1)[[2L]](list(a2, a2)),
  "linearize/vec_sin" = function(a0, a1) linearize(RF$vec_sin, a0)[[2L]](a1),
  "linearize_out/vec_sin" = function(a0) linearize(RF$vec_sin, a0)[[1L]],
  "linearize/two_out" = function(a0, a1, a2) linearize(RF$two_out, a0, a1)[[2L]](a2, a2),
  "vmap/dot_0_none" = function(a0, a1) vmap(RF$dot, in_axes = list(0L, NULL))(a0, a1),
  "vmap/dot_1_0" = function(a0, a1) vmap(RF$dot, in_axes = list(1L, 0L))(a0, a1),
  "vmap/dot_out1" = function(a0, a1) vmap(RF$dot, in_axes = list(NULL, 0L), out_axes = 0L)(a0, a1),
  "vmap/vec_sin_out1" = function(a0) vmap(RF$vec_sin, in_axes = 1L, out_axes = 1L)(a0),
  "vmap/vec_sin_neg" = function(a0) vmap(RF$vec_sin, in_axes = -1L, out_axes = -1L)(a0),
  "vmap/two_out" = function(a0, a1) vmap(RF$two_out, in_axes = list(0L, NULL), out_axes = list(0L, 1L))(a0, a1),
  "vmap/out_none" = function(a0, a1) {
    vmap(function(x, y) jnp_sum(y), in_axes = list(0L, NULL), out_axes = NULL)(a0, a1)
  },
  "vmap/dict" = function(a0) vmap(RF$dict_fn, in_axes = list(list(b = NULL, w = 0L)))(a0),
  "vmap/sum_axis" = function(a0) vmap(function(x) jnp_sum(x, axis = 0L), in_axes = 2L)(a0),
  "vmap/reduce_max" = function(a0) vmap(function(x) jnp_max(x, axis = 1L), in_axes = 1L)(a0),
  "vmap/transpose" = function(a0) vmap(function(x) jnp_transpose(x) * 2, in_axes = 1L, out_axes = 2L)(a0),
  "vmap/concat" = function(a0, a1) vmap(function(x, y) jnp_concatenate(list(x, y)), in_axes = list(0L, NULL))(a0, a1),
  "vmap/broadcast_add" = function(a0, a1) vmap(function(x, y) x + y, in_axes = list(1L, NULL))(a0, a1),
  "vmap/matmul_batched" = function(a0, a1) vmap(jnp_matmul, in_axes = list(0L, 2L))(a0, a1),
  "vmap/select" = function(a0, a1) vmap(function(c, x) jnp_where(c, x, -x), in_axes = list(NULL, 0L))(a0, a1),
  "vmap/convert" = function(a0) vmap(function(x) jnp_astype(x, "int32"), in_axes = 1L)(a0),
  "vmap/argmax" = function(a0) vmap(function(x) jnp_argmax(x), in_axes = 1L)(a0),
  "vmap/pad" = function(a0) vmap(function(x) lax_pad(x, 0, list(c(1L, 0L, 1L))))(a0),
  "vmap/slice" = function(a0) vmap(function(x) lax_slice(x, 1L, 3L), in_axes = 1L)(a0),
  "vmap/reshape" = function(a0) vmap(function(x) jnp_reshape(x, 4L), in_axes = 2L)(a0),
  "vmap/integer_pow" = function(a0) vmap(function(x) x^3, in_axes = 1L)(a0),
  "vmap/clamp" = function(a0, a1) vmap(function(lo, x) lax_clamp(lo, x, 1), in_axes = list(0L, 0L))(a0, a1),
  "vmap/dynamic_slice" = function(a0) vmap(function(x) lax_dynamic_slice(x, 1L, 2L))(a0),
  "vmap/rev" = function(a0) vmap(function(x) lax_rev(x, 0L), in_axes = 1L)(a0),
  "vmap/iota_add" = function(a0) vmap(function(x) x + jnp_arange(3, dtype = "float"))(a0),
  "vmap/axis_size" = function(a0) vmap(function(x) x * 2, in_axes = NULL, axis_size = 3L)(a0),
  "jit/sin_sq" = function(a0) jit(RF$sin_sq)(a0),
  "make_grad_grad/scalar_cube" = function(a0) grad(grad(RF$scalar_cube))(a0),
  "grad_grad_grad/scalar_cube" = function(a0) grad(grad(grad(RF$scalar_cube)))(a0),
  "vmap_grad/scalar_cube" = function(a0) vmap(grad(RF$scalar_cube))(a0),
  "vmap_grad/tanh_mv" = function(a0, a1) vmap(grad(RF$tanh_mv, argnums = 1L), in_axes = list(NULL, 0L))(a0, a1),
  "grad_vmap/vec_sin" = function(a0) grad(function(x) jnp_sum(vmap(RF$vec_sin)(x)))(a0),
  "grad_vmap/dot" = function(a0, a1) {
    grad(function(W, x) jnp_sum(vmap(RF$dot, in_axes = list(0L, NULL))(W, x)^2))(a0, a1)
  },
  "jit_grad/sin_sq" = function(a0) jit(grad(RF$sin_sq))(a0),
  "jit_grad/tanh_mv" = function(a0, a1) jit(grad(RF$tanh_mv, argnums = c(0L, 1L)))(a0, a1),
  "grad_jit/sin_sq" = function(a0) grad(jit(RF$sin_sq))(a0),
  "grad_jit/pow" = function(a0, a1) grad(jit(RF$pow), argnums = 1L)(a0, a1),
  "vmap_vmap/dot" = function(a0, a1) vmap(vmap(RF$dot, in_axes = list(0L, NULL)), in_axes = list(NULL, 0L))(a0, a1),
  "vmap_vmap/vec_sin" = function(a0) vmap(vmap(RF$vec_sin), in_axes = 1L, out_axes = 1L)(a0),
  "jvp_grad/sin_sq" = function(a0, a1) jvp(grad(RF$sin_sq), list(a0), list(a1)),
  "jvp_grad/scalar_cube" = function(a0, a1) jvp(grad(RF$scalar_cube), list(a0), list(a1)),
  "grad_jvp/vec_sin" = function(a0) grad(function(x) jnp_sum(jvp(RF$vec_sin, list(x), list(x))[[2L]]))(a0),
  "vmap_jvp/vec_sin" = function(a0, a1) vmap(function(x, t) jvp(RF$vec_sin, list(x), list(t))[[2L]])(a0, a1),
  "jit_vmap/dot" = function(a0, a1) jit(vmap(RF$dot, in_axes = list(0L, NULL)))(a0, a1),
  "vmap_jit/dot" = function(a0, a1) vmap(jit(RF$dot), in_axes = list(0L, NULL))(a0, a1),
  "linear_transpose/affine" = function(a0, a1) linear_transpose(function(x) 2 * x, a0)(a1),
  "linear_transpose/lin2" = function(a0, a1, a2, a3) linear_transpose(RF$lin2, a0, a1)(list(a2, a3)),
  "linear_transpose/matmul" = function(a0, a1, a2) linear_transpose(function(x) jnp_matmul(a1, x), a0)(a2),
  "linear_transpose/reshape_t" = function(a0, a1) {
    linear_transpose(function(x) jnp_transpose(jnp_reshape(x, c(3L, 2L))), a0)(a1)
  },
  "eval_shape/tanh_mv" = function(a0, a1) eval_shape(RF$tanh_mv, a0, a1),
  "eval_shape/two_out" = function(a0, a1) eval_shape(RF$two_out, a0, a1),
  "eval_shape/weak" = function(a0, a1) eval_shape(function(x, y) x * y + 1, a0, a1),
  "eval_shape/int" = function(a0) eval_shape(function(x) jnp_sum(x) %/% 2L, a0),
  "ravel_pytree/mixed" = function(a0) ravel_pytree(a0)$flat,
  "ravel_pytree/unravel" = function(a0) { r <- ravel_pytree(a0); r$unravel(r$flat * 2) },
  "ravel_pytree/list" = function(a0) ravel_pytree(a0)$flat,
  "grad_error/int_input" = function(a0) grad(function(x) jnp_sum(x * 1))(a0),
  "grad_error/non_scalar" = function(a0) grad(function(x) x * 2)(a0),
  "grad_error/int_output" = function(a0) grad(function(x) jnp_astype(jnp_sum(x), "int32"))(a0),
  "custom_jvp/value" = function(a0) cj_sin(a0),
  "custom_jvp/grad" = function(a0) grad(function(x) jnp_sum(cj_sin(x)))(a0),
  "custom_jvp/grad_grad" = function(a0) grad(grad(cj_sin))(a0),
  "custom_jvp/jvp" = function(a0, a1) jvp(cj_sin, list(a0), list(a1)),
  "custom_jvp/vmap" = function(a0) vmap(cj_sin)(a0),
  "custom_jvp/vmap_grad" = function(a0) vmap(grad(cj_sin))(a0),
  "custom_jvp/jit_grad" = function(a0) jit(grad(function(x) jnp_sum(cj_sin(x))))(a0),
  "custom_jvp/nondiff" = function(a0) grad(function(x) jnp_sum(cj_pow(3, x)))(a0),
  "custom_jvp/defjvps" = function(a0) grad(function(x) jnp_sum(cj_exp(x)))(a0),
  "custom_jvp/jacfwd" = function(a0) jacfwd(cj_sin)(a0),
  "custom_vjp/value" = function(a0) cv_sq(a0),
  "custom_vjp/grad" = function(a0) grad(cv_sq)(a0),
  "custom_vjp/grad_grad" = function(a0) grad(grad(cv_sq))(a0),
  "custom_vjp/jacfwd_jacrev" = function(a0) jacfwd(jacrev(cv_sq))(a0),
  "custom_vjp/jvp_error" = function(a0, a1) jvp(cv_sq, list(a0), list(a1)),
  "custom_vjp/vmap_grad" = function(a0) vmap(grad(cv_sq))(a0),
  "custom_vjp/jit_grad" = function(a0) jit(grad(function(x) jnp_sum(cv_sq(x))))(a0),
  "custom_vjp/two_args" = function(a0, a1) grad(function(x, y) jnp_sum(cv_two(x, y)), argnums = c(0L, 1L))(a0, a1),
  "custom_vjp/vmap" = function(a0) vmap(cv_sq)(a0),
  "stop_gradient/grad" = function(a0) grad(function(x) jnp_sum(x * lax_stop_gradient(x)))(a0),
  "stop_gradient/jvp" = function(a0, a1) jvp(function(x) x * lax_stop_gradient(x)^2, list(a0), list(a1)),
  "stop_gradient/int" = function(a0) lax_stop_gradient(a0),
  "cond/true" = function(a0) lax_cond(TRUE, function(x) x * 2, function(x) -x, a0),
  "cond/false" = function(a0) lax_cond(FALSE, function(x) x * 2, function(x) -x, a0),
  "cond/traced" = function(a0, a1, a2) lax_cond(a0 > 0, function(x, y) x * y, function(x, y) x - y, a1, a2),
  "cond/int_pred" = function(a0, a1) lax_cond(a0, function(x) x + 1, function(x) x - 1, a1),
  "cond/grad" = function(a0) grad(function(x) lax_cond(x > 0, jnp_sin, jnp_cos, x))(a0),
  "cond/grad_neg" = function(a0) grad(function(x) lax_cond(x > 0, jnp_sin, jnp_cos, x))(a0),
  "cond/vmap" = function(a0, a1) {
    vmap(function(p, x) lax_cond(p > 0, function(v) v * 2, function(v) v - 1, x))(a0, a1)
  },
  "cond/jvp" = function(a0, a1) jvp(function(x) lax_cond(jnp_sum(x) > 0, jnp_sin, jnp_cos, x), list(a0), list(a1)),
  "cond/pytree" = function(a0) lax_cond(TRUE, function(p) list(a = p$a * 2, b = p$b), function(p) p, a0),
  "cond/error_shape" = function(a0) lax_cond(TRUE, function(x) x, function(x) jnp_sum(x), a0),
  "switch/-1" = function(a0, a1) lax_switch(a0, SW, a1),
  "switch/0" = function(a0, a1) lax_switch(a0, SW, a1),
  "switch/1" = function(a0, a1) lax_switch(a0, SW, a1),
  "switch/2" = function(a0, a1) lax_switch(a0, SW, a1),
  "switch/7" = function(a0, a1) lax_switch(a0, SW, a1),
  "switch/grad" = function(a0) grad(function(x) lax_switch(1L, list(jnp_sin, jnp_cos, jnp_exp), x))(a0),
  "switch/vmap" = function(a0, a1) vmap(function(i, x) lax_switch(i, SW, x))(a0, a1),
  "while/basic" = function(a0, a1) {
    lax_while_loop(function(c) c[[1L]] < 5L, function(c) list(c[[1L]] + 1L, c[[2L]] * 1.5), list(a0, a1))
  },
  "while/jvp" = function(a0, a1) {
    jvp(function(x) lax_while_loop(function(c) c[[1L]] < 3L, function(c) list(c[[1L]] + 1L, jnp_sin(c[[2L]]) * 2),
                                   list(i32(0L), x))[[2L]], list(a0), list(a1))
  },
  "while/vmap" = function(a0, a1) {
    vmap(function(n, x) lax_while_loop(function(c) c[[1L]] < n, function(c) list(c[[1L]] + 1L, c[[2L]] * 2),
                                       list(i32(0L), x))[[2L]])(a0, a1)
  },
  "while/grad_error" = function(a0) {
    grad(function(x) lax_while_loop(function(c) c[[1L]] < 3L, function(c) list(c[[1L]] + 1L, c[[2L]] * 2),
                                    list(i32(0L), x))[[2L]])(a0)
  },
  "while/cond_error" = function(a0) lax_while_loop(function(c) c, function(c) c + 1, a0),
  "fori/static" = function(a0) lax_fori_loop(0L, 5L, function(i, v) v * 2 + i, a0),
  "fori/grad" = function(a0) grad(function(x) jnp_sum(lax_fori_loop(0L, 3L, function(i, v) jnp_sin(v) * v, x)))(a0),
  "fori/traced_bounds" = function(a0, a1) jit(function(n, x) lax_fori_loop(0L, n, function(i, v) v + 1.5, x))(a0, a1),
  "scan/cumsum" = function(a0) lax_scan(function(c, x) list(c + x, c * x), 0, a0),
  "scan/reverse" = function(a0, a1) lax_scan(function(c, x) list(c * 0.5 + x, c), a0, a1, reverse = TRUE),
  "scan/length" = function(a0) lax_scan(function(c, x) list(c * 2, c), a0, NULL, length = 4L),
  "scan/pytree" = function(a0, a1, a2, a3) {
    lax_scan(function(c, xy) list(list(c[[1L]] + xy[[1L]], c[[2L]] * xy[[2L]]), xy[[1L]] * c[[2L]]),
             list(a0, a1), list(a2, a3))
  },
  "scan/grad" = function(a0) {
    grad(function(x) jnp_sum(lax_scan(function(c, v) list(jnp_sin(c) + v, c * v), 0.5, x)[[2L]]))(a0)
  },
  "scan/vmap" = function(a0) vmap(function(xs) lax_scan(function(c, v) list(c + v, c * v), 0, xs))(a0),
  "scan/jvp" = function(a0, a1) jvp(function(x) lax_scan(function(c, v) list(c * v, c), 1, x), list(a0), list(a1)),
  "scan/jit_grad" = function(a0) jit(grad(function(x) lax_scan(function(c, v) list(c * v, NULL), 1, x)[[1L]]))(a0),
  "map/basic" = function(a0) lax_map(function(v) v^2 + 1, a0),
  "map/batch" = function(a0) lax_map(function(v) jnp_sum(v) * v, a0, batch_size = 2L),
  "map/pytree" = function(a0, a1) lax_map(function(p) p[[1L]] * p[[2L]], list(a0, a1)),
  "jacfwd/vec" = function(a0) jacfwd(JF)(a0),
  "jacrev/vec" = function(a0) jacrev(JF)(a0),
  "jacobian/vec" = function(a0) jacobian(JF)(a0),
  "jacrev/argnums" = function(a0, a1) jacrev(RF$two_out, argnums = c(0L, 1L))(a0, a1),
  "jacfwd/argnums1" = function(a0, a1) jacfwd(RF$two_out, argnums = 1L)(a0, a1),
  "jacrev/dict" = function(a0) jacrev(RF$dict_fn)(a0),
  "jacfwd/matrix" = function(a0, a1) jacfwd(function(W, x) jnp_tanh(jnp_matmul(W, x)))(a0, a1),
  "jacrev/has_aux" = function(a0) jacrev(function(x) list(jnp_sin(x), jnp_sum(x)), has_aux = TRUE)(a0),
  "jacfwd/has_aux" = function(a0) jacfwd(function(x) list(jnp_sin(x), jnp_sum(x)), has_aux = TRUE)(a0),
  "hessian/poly" = function(a0) hessian(function(x) jnp_sum(x^3) * jnp_sum(x))(a0),
  "hessian/sin_sq" = function(a0) hessian(RF$sin_sq)(a0),
  "hessian/scalar" = function(a0) hessian(RF$scalar_cube)(a0),
  "jacrev/scalar_in" = function(a0) jacrev(RF$vec_sin)(a0),
  "vmap_jacrev/vec_sin" = function(a0) vmap(jacrev(RF$vec_sin))(a0),
  "jit_hessian/sin_sq" = function(a0) jit(hessian(RF$sin_sq))(a0)
)

r_case_fun <- function(id) {
  if (!is.null(R_CASES[[id]])) return(R_CASES[[id]])
  kind <- sub("/.*$", "", id)
  f <- RF[[sub("^.*/", "", id)]]
  switch(kind,
    grad = function(...) grad(f)(...),
    grad_argnums0 = function(...) grad(f, argnums = 0L)(...),
    grad_argnums1 = function(...) grad(f, argnums = 1L)(...),
    "grad_argnums(0,1)" = function(...) grad(f, argnums = c(0L, 1L))(...),
    NULL)
}

transform_known <- character()

for (x64 in c(FALSE, TRUE)) local({
  x64 <- x64
  tag <- if (x64) "x64" else "x32"
  fx <- fx_load("transforms", x64)
  test_that(sprintf("every transform fixture has an R counterpart [%s]", tag), {
    ids <- vapply(fx$cases, `[[`, "", "id")
    expect_identical(ids[vapply(ids, function(i) is.null(r_case_fun(i)), NA)], character())
    expect_setequal(setdiff(names(fx$funs), names(R_FUNS)), character())
  })
  for (case in fx$cases) local({
    case <- case
    fun <- r_case_fun(case$id)
    if (is.null(fun)) return()
    reason <- fx_skip_reason(transform_known, case$id, tag)
    test_that(sprintf("%s matches the wheel [%s]", case$id, tag), {
      skip_if_no_backend()
      if (length(reason)) skip(reason[[1L]])
      with_jax_config("jax_enable_x64", x64, {
        args <- fx_args(case$args)
        info <- paste0(case$code, " [", tag, "]")
        if (!is.null(case$error)) {
          fx_expect_error(function() do.call(fun, args), case$error, info, check_message = FALSE)
          # jax.jit(jvp(custom_vjp)) also fails only at lowering (custom_lin has no lowering
          # rule in JAX either), so only the eager error class is compared for that case.
          if (case$id == "custom_vjp/jvp_error")
            expect_error(do.call(jit(fun), args), "custom_lin|forward-mode")
          else
            fx_expect_error(function() do.call(jit(fun), args), case$error, paste(info, "under jit"),
                            check_message = FALSE)
        } else {
          # Derivatives accumulate and cancel (tanh' near saturation amplifies one ulp of
          # tanh), so compare relative to the largest element; zero signs are not checked.
          cmp <- fx_cmp(ulps = 8, scale = TRUE, zero_sign = FALSE)
          fx_expect(do.call(fun, args), case$out, info, cmp)
          if (fx_jittable(case$out))
            fx_expect(do.call(jit(fun), args), case$out, paste(info, "under jit"), cmp)
        }
      })
    })
  })
})

# ---- behavior not covered by the fixtures ---------------------------------------------

test_that("gradients agree with central finite differences (float64)", {
  skip_if_no_backend()
  with_jax_config("jax_enable_x64", TRUE, {
    x <- as_jax(c(0.3, -0.7, 1.2), "float64")
    for (nm in c("sin_sq", "poly", "softplus", "logistic", "where", "concat_slice", "mean_var",
                 "rev_bcast", "exp_log", "cumul")) {
      xx <- if (nm == "exp_log") as_jax(c(0.3, 0.7, 1.2), "float64") else x
      check_grad_fd(RF[[nm]], xx)
    }
    check_grad_fd(function(x) RF$tanh_mv(as_jax(matrix(c(1, 0.5, -0.5, 2, 0.25, -1), 2), "float64"), x), x)
  })
})

test_that("vmap agrees with a stacked loop for assorted primitives", {
  skip_if_no_backend()
  m <- as_jax(matrix(c(0.5, -1, 2, 1.5, 0.25, -0.75), 2, byrow = TRUE))
  fs <- list(
    function(x) jnp_exp(x) * jnp_cos(x), function(x) jnp_sum(x^2),
    function(x) jnp_concatenate(list(x, x * 2)), function(x) lax_pad(x, 1, list(c(1L, 1L, 0L))),
    function(x) jnp_stack(list(x, -x)), function(x) jnp_where(x > 0, x, 0),
    function(x) jnp_matmul(x, as_jax(diag(3) * 2)), function(x) jnp_argmin(x),
    function(x) jnp_reshape(x, c(3L, 1L)), function(x) jnp_moveaxis(jnp_reshape(x, c(1L, 3L)), 0L, 1L))
  for (f in fs) {
    got <- vmap(f)(m)
    want <- jnp_stack(lapply(0:1, function(i) f(lax_index_in_dim(m, i, keepdims = FALSE))))
    expect_jax_equal(got, as.array(want))
  }
})

test_that("jit traces once per signature and retraces on new shapes, dtypes, weak types and statics", {
  skip_if_no_backend()
  n <- 0L
  f <- jit(function(x, k) { n <<- n + 1L; x * k }, static_argnums = 1L)
  f(as_jax(c(1, 2)), 2)
  f(as_jax(c(3, 4)), 2)
  expect_identical(n, 1L)
  expect_jax_equal(f(as_jax(c(3, 4)), 2), c(6, 8))
  f(as_jax(c(1, 2, 3)), 2)
  expect_identical(n, 2L)
  f(as_jax(c(1L, 2L)), 2)
  expect_identical(n, 3L)
  f(as_jax(c(1, 2)), 3)
  expect_identical(n, 4L)
  expect_jax_equal(f(as_jax(c(1, 2)), 3), c(3, 6))
  expect_identical(n, 4L)
  g <- jit(function(x) { n <<- n + 1L; x + 1 })
  n <- 0L
  g(2)
  g(as_jax(2, "float32"))
  expect_identical(n, 2L)
  g(3)
  expect_identical(n, 2L)
  with_jax_config("jax_enable_x64", TRUE, g(3))
  expect_identical(n, 3L)
})

test_that("jit static_argnames and keyword arguments", {
  skip_if_no_backend()
  n <- 0L
  f <- jit(function(x, power = 2L) { n <<- n + 1L; x^power }, static_argnames = "power")
  expect_jax_equal(f(as_jax(c(1, 2)), power = 3L), c(1, 8))
  expect_jax_equal(f(as_jax(c(2, 3)), power = 3L), c(8, 27))
  expect_identical(n, 1L)
  expect_jax_equal(f(as_jax(c(2, 3)), power = 2L), c(4, 9))
  expect_identical(n, 2L)
  h <- jit(function(x, y) x - y)
  expect_jax_equal(h(y = as_jax(1), x = as_jax(5)), 4)
  expect_jax_equal(h(as_jax(5), as_jax(1)), 4)
})

test_that("jit handles pytree arguments and outputs and disable_jit", {
  skip_if_no_backend()
  f <- jit(function(p) list(s = p$a + p$b, d = p$a - p$b))
  out <- f(list(a = as_jax(c(1, 2)), b = as_jax(c(3, 5))))
  expect_named(out, c("s", "d"))
  expect_jax_equal(out$s, c(4, 7))
  expect_jax_equal(out$d, c(-2, -3))
  seen <- NULL
  g <- jit(function(x) { seen <<- class(x)[1L]; x * 2 })
  disable_jit(g(as_jax(1)))
  expect_identical(seen, "jax_array")
  g(as_jax(1))
  expect_identical(seen, "jax_tracer")
  expect_s3_class(g, "jax_jitted")
})

test_that("make_jaxpr records primitives, avals and static arguments", {
  skip_if_no_backend()
  cj <- make_jaxpr(function(x, y) jnp_sin(x) * y)(as_jax(c(1, 2, 3)), as_jax(c(4, 5, 6)))
  expect_s3_class(cj, "jax_closed_jaxpr")
  prims <- vapply(cj$jaxpr$eqns, function(e) e$prim$name, "")
  expect_identical(prims, c("sin", "mul"))
  expect_identical(lapply(cj$jaxpr$invars, function(v) format(v$aval)), list("f32[3]", "f32[3]"))
  expect_output(print(cj), "\\{ lambda ; a:f32\\[3\\] b:f32\\[3\\]\\. let")
  expect_output(print(cj), "c:f32\\[3\\] = sin a")
  cj2 <- make_jaxpr(function(x, n) jnp_sum(x, axis = n), static_argnums = 1L)(as_jax(matrix(1:6, 2)), 1L)
  expect_identical(vapply(cj2$jaxpr$eqns, function(e) e$prim$name, ""), "reduce_sum")
  expect_identical(cj2$jaxpr$eqns[[1L]]$params$axes, 1L)
  expect_identical(format(cj2$jaxpr$outvars[[1L]]$aval), "i32[2]")
  cj3 <- make_jaxpr(grad(function(x) jnp_sum(jnp_sin(x))))(as_jax(c(1, 2)))
  expect_true("cos" %in% vapply(cj3$jaxpr$eqns, function(e) e$prim$name, ""))
})

test_that("make_jaxpr folds weak scalar constants like JAX", {
  skip_if_no_backend()
  cj <- make_jaxpr(function(x) jnp_sin(x) * 2)(as_jax(c(1, 2, 3)))
  expect_identical(vapply(cj$jaxpr$eqns, function(e) e$prim$name, ""), c("sin", "mul"))
})

test_that("vmap validates in_axes, out_axes and sizes like jax.vmap", {
  skip_if_no_backend()
  x <- as_jax(matrix(1:6, 2))
  expect_error(vmap(function(a, b) a + b)(x, as_jax(1:3)), "inconsistent sizes")
  expect_error(vmap(function(a) a, in_axes = 2L)(x), "rank should be at least 3")
  expect_error(vmap(function(a, b) a, in_axes = list(0L))(x, x), "one entry per positional argument")
  expect_error(vmap(function(a) a, out_axes = NULL)(x), "out_axes is None")
  expect_error(vmap(function(a) a, in_axes = NULL)(x), "at least one non-None")
})

test_that("vmap matches dict in_axes by name, not position", {
  skip_if_no_backend()
  p <- list(b = as_jax(c(0.5, 3)), w = as_jax(matrix(c(1, 0.5, -2, 0.25), 2)))
  want <- vmap(RF$dict_fn, in_axes = list(list(b = NULL, w = 0L)))(p)
  got <- vmap(RF$dict_fn, in_axes = list(list(w = 0L, b = NULL)))(p)
  expect_jax_equal(got, as.array(want))
})

test_that("grad, jvp and linearize validate their arguments", {
  skip_if_no_backend()
  f <- function(x, y) jnp_sum(x * y)
  expect_error(grad(f, argnums = 2L)(as_jax(1), as_jax(2)),
               "argnums=2 requires at least 3 positional arguments", class = "jax_type_error")
  expect_error(jvp(f, as_jax(1), list(as_jax(1))), "must be lists", class = "jax_type_error")
  expect_error(jvp(function(x) x, list(as_jax(c(1, 2))), list(as_jax(c(1, 2, 3)))),
               "different primal and tangent shapes")
  # The tangent of an integer primal is float0.
  expect_error(jvp(function(x, n) x * n, list(as_jax(c(1, 2)), as_jax(3L)), list(as_jax(c(1, 1)), as_jax(0L))),
               "tangent dtype must be float0", class = "jax_type_error")
  r <- jvp(function(x, n) x * n, list(as_jax(c(1, 2)), as_jax(3L)),
           list(as_jax(c(1, 1)), jnp_zeros(integer(0), "float0")))
  expect_jax_equal(r[[2L]], c(3, 3))
})

test_that("vjp and grad give zero cotangents for integer and unused inputs", {
  skip_if_no_backend()
  r <- vjp(function(x, n) x * 2, as_jax(c(1, 2)), as_jax(3L))
  ct <- r[[2L]](as_jax(c(1, 1)))
  expect_jax_equal(ct[[1L]], c(2, 2))
  expect_identical(dtype(ct[[2L]]), "float0")
  g <- grad(function(x, y) jnp_sum(x), argnums = c(0L, 1L))(as_jax(c(1, 2)), as_jax(3))
  expect_jax_equal(g[[2L]], 0)
})

test_that("linear_transpose of a non-linear function errors", {
  skip_if_no_backend()
  expect_error(linear_transpose(function(x) x * x, as_jax(c(1, 2)))(as_jax(c(1, 1))), "both arguments are linear")
})

test_that("has_aux outputs of vjp and jvp are values, not leaked tracers", {
  skip_if_no_backend()
  x <- as_jax(c(1, 2))
  r <- vjp(function(x) list(jnp_sin(x), x * 2), x, has_aux = TRUE)
  expect_jax_equal(r[[3L]], c(2, 4))
  r <- jvp(function(x) list(jnp_sin(x), x * 2), list(x), list(x), has_aux = TRUE)
  expect_jax_equal(r[[3L]], c(2, 4))
})

test_that("control flow validates its arguments like jax.lax", {
  skip_if_no_backend()
  expect_error(lax_cond(as_jax(c(TRUE, FALSE)), function(x) x, function(x) x, 1), class = "jax_type_error")
  expect_error(lax_switch(0L, list(), 1), "Empty branch sequence")
  expect_error(lax_switch(0.5, list(function(x) x, function(x) x), 1), "Index type must be an integer")
  expect_error(lax_while_loop(function(c) c < 3L, function(c) c + 0.5, as_jax(0L, "int32")),
               class = "jax_type_error")
  expect_error(lax_scan(function(c, x) list(c, x), 0, list(as_jax(1:3), as_jax(1:2))),
               "different leading axis sizes")
  expect_error(lax_scan(function(c, x) list(c, c), 0), "no values to scan over")
  expect_error(lax_scan(function(c, x) c, 0, as_jax(1:3)), "must return a list")
  expect_error(lax_scan(function(c, x) list(list(c), x), 0, as_jax(1:3)), class = "jax_type_error")
  expect_error(lax_fori_loop(0, 2.5, function(i, v) v, 1), "integer dtypes")
})

test_that("disable_jit runs control flow as R control flow", {
  skip_if_no_backend()
  seen <- character()
  disable_jit({
    r <- lax_cond(TRUE, function(x) { seen <<- c(seen, "t"); x * 2 }, function(x) { seen <<- c(seen, "f"); x }, as_jax(3))
    w <- lax_while_loop(function(c) c < 3L, function(c) c + 1L, as_jax(0L, "int32"))
  })
  expect_identical(seen, "t")
  expect_jax_equal(r, 6)
  expect_jax_equal(w, 3)
})

test_that("control flow results match a plain R loop", {
  skip_if_no_backend()
  xs <- as_jax(c(0.5, -1, 2, 0.25))
  r <- lax_scan(function(c, x) list(c * 0.5 + x, c^2), as_jax(1, "float32"), xs)
  carry <- 1; ys <- numeric(0)
  for (x in c(0.5, -1, 2, 0.25)) { ys <- c(ys, carry^2); carry <- carry * 0.5 + x }
  expect_jax_equal(r[[1L]], carry)
  expect_jax_equal(r[[2L]], ys)
  expect_jax_equal(lax_map(function(v) v * 2, xs), c(1, -2, 4, 0.5))
  expect_jax_equal(lax_fori_loop(1L, 4L, function(i, v) v + i, as_jax(0L, "int32")), 6)
  expect_jax_equal(jit(function(n) lax_fori_loop(0L, n, function(i, v) v + i, as_jax(0L, "int32")))(as_jax(5L, "int32")), 10)
})

test_that("scan keeps the weak type of a weak init carry in ys, like JAX", {
  skip_if_no_backend()
  r <- lax_scan(function(c, x) list(c + x, c), 0, as_jax(c(1, 2, 3)))
  expect_false(jaxR:::aval_of(r[[1L]])$weak)
  expect_true(jaxR:::aval_of(r[[2L]])$weak)
  # A weak init of another dtype is promoted to the dtype the body returns.
  r <- lax_scan(function(c, x) list(c + x, c), 0L, as_jax(c(1, 2, 3)))
  expect_identical(jaxR:::aval_of(r[[1L]])$dtype, "float32")
  expect_equal(as.vector(as.array(r[[2L]])), c(0, 1, 3))
  w <- lax_while_loop(function(c) c < 3, function(c) c + as_jax(1, "float32"), 0L)
  expect_identical(jaxR:::aval_of(w)$dtype, "float32")
  expect_equal(as.vector(as.array(w)), 3)
})

test_that("design gates: custom_vjp and gather gradients", {
  skip_if_no_backend()
  f <- custom_vjp(function(x) x^2, fwd = function(x) list(x^2, x), bwd = function(res, ct) list(3 * res * ct))
  x <- as_jax(2)
  expect_jax_equal(grad(f)(x), 6)
  expect_jax_equal(grad(grad(f))(x), 3)
  expect_jax_equal(jacfwd(jacrev(f))(x), 3)
  expect_error(jvp(f, list(x), list(x)), class = "jax_type_error")
  expect_jax_equal(grad(function(v) jnp_sum(v[c(0L, 0L, 2L)]))(as_jax(c(1, 2, 3))), c(2, 0, 1))
})
