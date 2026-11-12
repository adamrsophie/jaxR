# jax.example_libraries.stax (port of jax/example_libraries/stax.py).
#
# A layer is a pair list(init_fun, apply_fun):
#   init_fun(rng, input_shape) -> list(output_shape, params)
#   apply_fun(params, inputs, ...) -> outputs (an `rng` key may be passed as `rng = `)
# Shapes are integer vectors whose batch entry may be -1; parameterless layers have
# params list().

stx_layer <- function(init_fun, apply_fun) list(init_fun = init_fun, apply_fun = apply_fun)
stx_result <- function(output_shape, params) list(output_shape = output_shape, params = params)

# spec.index(c) of a layout string such as "NHWC" (0-based).
stx_index <- function(spec, ch) {
  i <- match(ch, strsplit(spec, "")[[1L]])
  if (is.na(i)) jax_error("'", ch, "' is not in the layout '", spec, "'", class = "jax_value_error")
  i - 1L
}

stx_kernel_shape <- function(dimension_numbers, out_chan, filter_shape, input_shape) {
  rhs <- strsplit(dimension_numbers[[2L]], "")[[1L]]
  in_chan <- input_shape[stx_index(dimension_numbers[[1L]], "C") + 1L]
  out <- integer(length(rhs))
  out[rhs == "O"] <- out_chan
  out[rhs == "I"] <- in_chan
  out[!rhs %in% c("O", "I")] <- filter_shape
  as.integer(out)
}

stx_bias_shape <- function(out_spec, out_chan) {
  as.integer(ifelse(strsplit(out_spec, "")[[1L]] == "C", out_chan, 1L))
}

#' Stax layers
#'
#' Ports of the layers of `jax.example_libraries.stax`. A layer is a pair
#' `list(init_fun, apply_fun)`: `init_fun(rng, input_shape)` returns
#' `list(output_shape, params)` and `apply_fun(params, inputs, ...)` applies the layer
#' (`stax_Dropout` needs a key passed as `rng = key`). Shapes are integer vectors whose
#' batch entry may be `-1`; parameterless layers have params `list()`.
#'
#' `stax_Dense` is a fully connected layer `inputs %*% W + b`. `stax_GeneralConv` and
#' `stax_GeneralConvTranspose` are (transposed) convolutions with layouts given as three
#' strings such as `c("NHWC", "HWIO", "NHWC")`; `stax_Conv`, `stax_ConvTranspose` and
#' `stax_Conv1DTranspose` fix the layouts to `NHWC`/`HWIO` and `NHC`/`HIO`.
#' `stax_BatchNorm` standardizes over `axis`. `stax_MaxPool`, `stax_SumPool` and
#' `stax_AvgPool` pool over windows of the spatial dimensions (the dimensions other than
#' `N` and `C` of `spec`, by default the first and last). `stax_elementwise` makes a layer
#' from an elementwise function; `stax_Tanh`, `stax_Relu`, `stax_Exp`, `stax_LogSoftmax`,
#' `stax_Softmax`, `stax_Softplus`, `stax_Sigmoid`, `stax_Elu`, `stax_LeakyRelu`,
#' `stax_Selu`, `stax_Gelu`, `stax_Flatten`, `stax_Identity` and `stax_FanInSum` are
#' layers. `stax_FanOut` copies its input `num` times, `stax_FanInSum` adds a list of
#' inputs and `stax_FanInConcat` concatenates them along `axis`.
#' @param out_dim Output dimension of `stax_Dense`.
#' @param W_init,b_init Initializers of the weights and biases (see [nn_initializers]).
#'   The default `W_init` of the convolutions is a Glorot normal initializer over the
#'   kernel's `I` and `O` axes.
#' @param dimension_numbers Three layout strings (input, kernel, output).
#' @param out_chan Number of output channels.
#' @param filter_shape Integer vector of kernel spatial sizes.
#' @param strides Integer vector of strides (default 1).
#' @param padding `"VALID"`, `"SAME"`, `"SAME_LOWER"` or a list of `c(low, high)` pairs.
#' @param axis 0-based axes standardized by `stax_BatchNorm`; axis of `stax_FanInConcat`.
#' @param epsilon Variance correction of `stax_BatchNorm`.
#' @param center,scale Whether `stax_BatchNorm` has a bias (`beta`) and a scale (`gamma`).
#' @param beta_init,gamma_init Initializers of `beta` and `gamma`.
#' @param fun Elementwise function.
#' @param ... Further arguments passed to `fun` by `stax_elementwise`.
#' @param window_shape Integer vector of window sizes over the spatial dimensions.
#' @param spec Optional layout string such as `"NHWC"` naming the batch and channel
#'   dimensions of the pooling input.
#' @param num Number of copies made by `stax_FanOut`.
#' @param rate Keep probability of `stax_Dropout` (kept values are divided by `rate`).
#' @param mode `"train"` applies dropout; any other value returns the inputs.
#' @return A layer `list(init_fun, apply_fun)`.
#' @examples
#' \dontrun{
#' layer <- stax_serial(stax_Dense(4L), stax_Relu, stax_Dense(1L))
#' res <- layer$init_fun(random_key(0), c(-1L, 3L))
#' layer$apply_fun(res$params, jnp_ones(c(2, 3)))
#' }
#' @name stax_layers
NULL

#' @rdname stax_layers
#' @export
stax_Dense <- function(out_dim, W_init = nn_initializers_glorot_normal(),
                       b_init = nn_initializers_normal()) {
  out_dim <- as.integer(out_dim)
  force(W_init); force(b_init)
  stx_layer(
    function(rng, input_shape) {
      input_shape <- as.integer(input_shape)
      ks <- nni_split(rng)
      w <- W_init(ks[[1L]], c(input_shape[length(input_shape)], out_dim))
      b <- b_init(ks[[2L]], out_dim)
      stx_result(c(utils::head(input_shape, -1L), out_dim), list(w, b))
    },
    function(params, inputs, ...) jnp_dot(inputs, params[[1L]]) + params[[2L]])
}

#' @rdname stax_layers
#' @export
stax_GeneralConv <- function(dimension_numbers, out_chan, filter_shape, strides = NULL,
                             padding = "VALID", W_init = NULL,
                             b_init = nn_initializers_normal(1e-6)) {
  out_chan <- as.integer(out_chan)
  filter_shape <- as.integer(filter_shape)
  one <- rep(1L, length(filter_shape))
  if (!length(strides)) strides <- one
  rhs_spec <- dimension_numbers[[2L]]
  if (is.null(W_init))
    W_init <- nn_initializers_glorot_normal(stx_index(rhs_spec, "I"), stx_index(rhs_spec, "O"))
  force(b_init); force(padding)
  stx_layer(
    function(rng, input_shape) {
      kernel_shape <- stx_kernel_shape(dimension_numbers, out_chan, filter_shape, input_shape)
      output_shape <- lax_conv_general_shape_tuple(input_shape, kernel_shape, strides, padding,
                                                   dimension_numbers)
      ks <- nni_split(rng)
      bias_shape <- stx_bias_shape(dimension_numbers[[3L]], out_chan)
      stx_result(output_shape, list(W_init(ks[[1L]], kernel_shape), b_init(ks[[2L]], bias_shape)))
    },
    function(params, inputs, ...) {
      lax_conv_general_dilated(inputs, params[[1L]], strides, padding, one, one,
                               dimension_numbers = dimension_numbers) + params[[2L]]
    })
}

#' @rdname stax_layers
#' @export
stax_Conv <- function(out_chan, filter_shape, strides = NULL, padding = "VALID", W_init = NULL,
                      b_init = nn_initializers_normal(1e-6)) {
  stax_GeneralConv(c("NHWC", "HWIO", "NHWC"), out_chan, filter_shape, strides, padding,
                   W_init, b_init)
}

#' @rdname stax_layers
#' @export
stax_GeneralConvTranspose <- function(dimension_numbers, out_chan, filter_shape, strides = NULL,
                                      padding = "VALID", W_init = NULL,
                                      b_init = nn_initializers_normal(1e-6)) {
  out_chan <- as.integer(out_chan)
  filter_shape <- as.integer(filter_shape)
  one <- rep(1L, length(filter_shape))
  if (!length(strides)) strides <- one
  rhs_spec <- dimension_numbers[[2L]]
  if (is.null(W_init))
    W_init <- nn_initializers_glorot_normal(stx_index(rhs_spec, "I"), stx_index(rhs_spec, "O"))
  force(b_init); force(padding)
  stx_layer(
    function(rng, input_shape) {
      kernel_shape <- stx_kernel_shape(dimension_numbers, out_chan, filter_shape, input_shape)
      output_shape <- lax_conv_transpose_shape_tuple(input_shape, kernel_shape, strides,
                                                     padding, dimension_numbers)
      ks <- nni_split(rng)
      bias_shape <- stx_bias_shape(dimension_numbers[[3L]], out_chan)
      stx_result(output_shape, list(W_init(ks[[1L]], kernel_shape), b_init(ks[[2L]], bias_shape)))
    },
    function(params, inputs, ...) {
      lax_conv_transpose(inputs, params[[1L]], strides, padding,
                         dimension_numbers = dimension_numbers) + params[[2L]]
    })
}

#' @rdname stax_layers
#' @export
stax_Conv1DTranspose <- function(out_chan, filter_shape, strides = NULL, padding = "VALID",
                                 W_init = NULL, b_init = nn_initializers_normal(1e-6)) {
  stax_GeneralConvTranspose(c("NHC", "HIO", "NHC"), out_chan, filter_shape, strides, padding,
                            W_init, b_init)
}

#' @rdname stax_layers
#' @export
stax_ConvTranspose <- function(out_chan, filter_shape, strides = NULL, padding = "VALID",
                               W_init = NULL, b_init = nn_initializers_normal(1e-6)) {
  stax_GeneralConvTranspose(c("NHWC", "HWIO", "NHWC"), out_chan, filter_shape, strides,
                            padding, W_init, b_init)
}

#' @rdname stax_layers
#' @export
stax_BatchNorm <- function(axis = c(0L, 1L, 2L), epsilon = 1e-5, center = TRUE, scale = TRUE,
                           beta_init = nn_initializers_zeros, gamma_init = nn_initializers_ones) {
  axis <- as.integer(axis)
  force(epsilon); force(center); force(scale); force(beta_init); force(gamma_init)
  stx_layer(
    function(rng, input_shape) {
      shape <- input_shape[!(seq_along(input_shape) - 1L) %in% axis]
      ks <- nni_split(rng)
      beta <- if (center) beta_init(ks[[1L]], shape) else list()
      gamma <- if (scale) gamma_init(ks[[2L]], shape) else list()
      stx_result(input_shape, list(beta, gamma))
    },
    function(params, x, ...) {
      x <- as_jax(x)
      nd <- length(dim(x))
      # gamma[ed] with None at the positions in `axis`.
      ed <- which((seq_len(nd) - 1L) %in% axis) - 1L
      z <- nn_standardize(x, axis, epsilon = epsilon)
      if (center && scale) return(lax_expand_dims(params[[2L]], ed) * z +
                                    lax_expand_dims(params[[1L]], ed))
      if (center) return(z + lax_expand_dims(params[[1L]], ed))
      if (scale) return(lax_expand_dims(params[[2L]], ed) * z)
      z
    })
}

#' @rdname stax_layers
#' @export
stax_elementwise <- function(fun, ...) {
  fun_kwargs <- list(...)
  force(fun)
  stx_layer(function(rng, input_shape) stx_result(input_shape, list()),
            function(params, inputs, ...) do.call(fun, c(list(inputs), fun_kwargs)))
}

#' @rdname stax_layers
#' @export
stax_Tanh <- stax_elementwise(jnp_tanh)
#' @rdname stax_layers
#' @export
stax_Relu <- stax_elementwise(nn_relu)
#' @rdname stax_layers
#' @export
stax_Exp <- stax_elementwise(jnp_exp)
#' @rdname stax_layers
#' @export
stax_LogSoftmax <- stax_elementwise(nn_log_softmax, axis = -1L)
#' @rdname stax_layers
#' @export
stax_Softmax <- stax_elementwise(nn_softmax, axis = -1L)
#' @rdname stax_layers
#' @export
stax_Softplus <- stax_elementwise(nn_softplus)
#' @rdname stax_layers
#' @export
stax_Sigmoid <- stax_elementwise(nn_sigmoid)
#' @rdname stax_layers
#' @export
stax_Elu <- stax_elementwise(nn_elu)
#' @rdname stax_layers
#' @export
stax_LeakyRelu <- stax_elementwise(nn_leaky_relu)
#' @rdname stax_layers
#' @export
stax_Selu <- stax_elementwise(nn_selu)
#' @rdname stax_layers
#' @export
stax_Gelu <- stax_elementwise(nn_gelu)

# Port of stax._pooling_layer.
stx_pooling_layer <- function(reducer, init_val, rescaler = NULL) {
  force(reducer); force(init_val); force(rescaler)
  function(window_shape, strides = NULL, padding = "VALID", spec = NULL) {
    window_shape <- as.integer(window_shape)
    force(padding); force(spec)
    strides <- if (length(strides)) as.integer(strides) else rep(1L, length(window_shape))
    rescale <- if (is.null(rescaler)) NULL else rescaler(window_shape, strides, padding)
    non_spatial_axes <- if (is.null(spec)) c(0L, length(window_shape) + 1L)
                        else c(stx_index(spec, "N"), stx_index(spec, "C"))
    for (i in sort(non_spatial_axes)) {
      window_shape <- append(window_shape, 1L, after = i)
      strides <- append(strides, 1L, after = i)
    }
    stx_layer(
      function(rng, input_shape) {
        padding_vals <- lax_padtype_to_pads(input_shape, window_shape, strides, padding)
        ones <- rep(1L, length(window_shape))
        stx_result(lax_reduce_window_shape_tuple(input_shape, window_shape, strides,
                                                 padding_vals, ones, ones), list())
      },
      function(params, inputs, ...) {
        out <- lax_reduce_window(inputs, init_val, reducer, window_shape, strides, padding)
        if (is.null(rescale)) out else rescale(out, inputs, spec)
      })
  }
}

# Port of stax._normalize_by_window_size: divide by the number of elements in each
# (padded) window.
stx_normalize_by_window_size <- function(dims, strides, padding) {
  force(dims); force(strides); force(padding)
  function(outputs, inputs, spec) {
    inputs <- as_jax(inputs)
    s <- dim(inputs)
    non_spatial_axes <- if (is.null(spec)) c(0L, length(s) - 1L)
                        else c(stx_index(spec, "N"), stx_index(spec, "C"))
    spatial_shape <- s[!(seq_along(s) - 1L) %in% non_spatial_axes]
    one <- jnp_ones(spatial_shape, dtype = dtype(inputs))
    window_sizes <- lax_reduce_window(one, 0, lax_add, dims, strides, padding)
    for (i in sort(non_spatial_axes)) window_sizes <- jnp_expand_dims(window_sizes, i)
    outputs / window_sizes
  }
}

#' @rdname stax_layers
#' @export
stax_MaxPool <- stx_pooling_layer(lax_max, -Inf)
#' @rdname stax_layers
#' @export
stax_SumPool <- stx_pooling_layer(lax_add, 0)
#' @rdname stax_layers
#' @export
stax_AvgPool <- stx_pooling_layer(lax_add, 0, stx_normalize_by_window_size)

#' @rdname stax_layers
#' @export
stax_Flatten <- stx_layer(
  function(rng, input_shape) {
    stx_result(as.integer(c(input_shape[1L], prod(drop_first(input_shape, 1L)))), list())
  },
  function(params, inputs, ...) jnp_reshape(inputs, c(dim(as_jax(inputs))[1L], -1L)))

#' @rdname stax_layers
#' @export
stax_Identity <- stx_layer(function(rng, input_shape) stx_result(input_shape, list()),
                           function(params, inputs, ...) inputs)

#' @rdname stax_layers
#' @export
stax_FanOut <- function(num) {
  num <- as.integer(num)
  stx_layer(function(rng, input_shape) stx_result(rep(list(input_shape), num), list()),
            function(params, inputs, ...) rep(list(inputs), num))
}

#' @rdname stax_layers
#' @export
stax_FanInSum <- stx_layer(function(rng, input_shape) stx_result(input_shape[[1L]], list()),
                           function(params, inputs, ...) Reduce(`+`, inputs, 0L))

#' @rdname stax_layers
#' @export
stax_FanInConcat <- function(axis = -1L) {
  axis <- as.integer(axis)
  stx_layer(
    function(rng, input_shape) {
      first <- as.integer(input_shape[[1L]])
      ax <- axis %% length(first)
      first[ax + 1L] <- sum(vapply(input_shape, function(s) as.numeric(s[ax + 1L]), 1))
      stx_result(first, list())
    },
    function(params, inputs, ...) jnp_concatenate(inputs, axis))
}

#' @rdname stax_layers
#' @export
stax_Dropout <- function(rate, mode = "train") {
  force(rate); force(mode)
  stx_layer(
    function(rng, input_shape) stx_result(input_shape, list()),
    function(params, inputs, ...) {
      rng <- list(...)[["rng"]]
      if (is.null(rng))
        jax_error("Dropout layer requires apply_fun to be called with a PRNG key argument. ",
                  "That is, instead of `apply_fun(params, inputs)`, call it like ",
                  "`apply_fun(params, inputs, rng = key)` where `key` is a PRNG key (e.g. ",
                  "from `random_key()`).", class = "jax_value_error")
      if (identical(mode, "train")) {
        inputs <- as_jax(inputs)
        keep <- random_bernoulli(rng, rate, dim(inputs))
        return(jnp_where(keep, inputs / rate, 0L))
      }
      inputs
    })
}

# ---- combinators ----------------------------------------------------------------------

#' Stax combinators
#'
#' Ports of `stax.serial`, `stax.parallel` and `stax.shape_dependent`. `stax_serial`
#' composes layers in sequence (its params are the list of the layers' params);
#' `stax_parallel` applies one layer to each element of a list of inputs (often between
#' [stax_FanOut()] and [stax_FanInSum]); `stax_shape_dependent` delays building a layer
#' until the input shape is known. A key passed to `apply_fun` as `rng = key` is split
#' between the layers.
#' @param ... Layers, each a pair `list(init_fun, apply_fun)`.
#' @param make_layer Function of an input shape returning a layer.
#' @return A layer `list(init_fun, apply_fun)`.
#' @examples
#' \dontrun{
#' net <- stax_serial(stax_FanOut(2L), stax_parallel(stax_Dense(3L), stax_Identity),
#'                    stax_FanInSum)
#' res <- net$init_fun(random_key(1), c(-1L, 3L))
#' }
#' @name stax_combinators
NULL

# Split an optional key into one key (or NULL) per layer.
stx_layer_rngs <- function(rng, n) if (is.null(rng)) vector("list", n) else nni_split(rng, n)

#' @rdname stax_combinators
#' @export
stax_serial <- function(...) {
  layers <- list(...)
  nlayers <- length(layers)
  init_funs <- lapply(layers, `[[`, 1L)
  apply_funs <- lapply(layers, `[[`, 2L)
  stx_layer(
    function(rng, input_shape) {
      params <- vector("list", nlayers)
      for (k in seq_len(nlayers)) {
        ks <- nni_split(rng)
        rng <- ks[[1L]]
        res <- init_funs[[k]](ks[[2L]], input_shape)
        input_shape <- res[[1L]]
        params[[k]] <- res[[2L]]
      }
      stx_result(input_shape, params)
    },
    function(params, inputs, ...) {
      kwargs <- list(...)
      rngs <- stx_layer_rngs(kwargs[["rng"]], nlayers)
      kwargs[["rng"]] <- NULL
      for (k in seq_len(nlayers))
        inputs <- do.call(apply_funs[[k]], c(list(params[[k]], inputs, rng = rngs[[k]]), kwargs))
      inputs
    })
}

#' @rdname stax_combinators
#' @export
stax_parallel <- function(...) {
  layers <- list(...)
  nlayers <- length(layers)
  init_funs <- lapply(layers, `[[`, 1L)
  apply_funs <- lapply(layers, `[[`, 2L)
  stx_layer(
    function(rng, input_shape) {
      rngs <- nni_split(rng, nlayers)
      res <- Map(function(f, r, s) f(r, s), init_funs, rngs, input_shape)
      stx_result(lapply(res, `[[`, 1L), lapply(res, `[[`, 2L))
    },
    function(params, inputs, ...) {
      kwargs <- list(...)
      rngs <- stx_layer_rngs(kwargs[["rng"]], nlayers)
      kwargs[["rng"]] <- NULL
      lapply(seq_len(nlayers), function(k) {
        do.call(apply_funs[[k]], c(list(params[[k]], inputs[[k]], rng = rngs[[k]]), kwargs))
      })
    })
}

#' @rdname stax_combinators
#' @export
stax_shape_dependent <- function(make_layer) {
  force(make_layer)
  stx_layer(
    function(rng, input_shape) make_layer(input_shape)[[1L]](rng, input_shape),
    function(params, inputs, ...) make_layer(dim(as_jax(inputs)))[[2L]](params, inputs, ...))
}
