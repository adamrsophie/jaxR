# Stax layers

Ports of the layers of `jax.example_libraries.stax`. A layer is a pair
`list(init_fun, apply_fun)`: `init_fun(rng, input_shape)` returns
`list(output_shape, params)` and `apply_fun(params, inputs, ...)`
applies the layer (`stax_Dropout` needs a key passed as `rng = key`).
Shapes are integer vectors whose batch entry may be `-1`; parameterless
layers have params [`list()`](https://rdrr.io/r/base/list.html).

## Usage

``` r
stax_Dense(
  out_dim,
  W_init = nn_initializers_glorot_normal(),
  b_init = nn_initializers_normal()
)

stax_GeneralConv(
  dimension_numbers,
  out_chan,
  filter_shape,
  strides = NULL,
  padding = "VALID",
  W_init = NULL,
  b_init = nn_initializers_normal(1e-06)
)

stax_Conv(
  out_chan,
  filter_shape,
  strides = NULL,
  padding = "VALID",
  W_init = NULL,
  b_init = nn_initializers_normal(1e-06)
)

stax_GeneralConvTranspose(
  dimension_numbers,
  out_chan,
  filter_shape,
  strides = NULL,
  padding = "VALID",
  W_init = NULL,
  b_init = nn_initializers_normal(1e-06)
)

stax_Conv1DTranspose(
  out_chan,
  filter_shape,
  strides = NULL,
  padding = "VALID",
  W_init = NULL,
  b_init = nn_initializers_normal(1e-06)
)

stax_ConvTranspose(
  out_chan,
  filter_shape,
  strides = NULL,
  padding = "VALID",
  W_init = NULL,
  b_init = nn_initializers_normal(1e-06)
)

stax_BatchNorm(
  axis = c(0L, 1L, 2L),
  epsilon = 1e-05,
  center = TRUE,
  scale = TRUE,
  beta_init = nn_initializers_zeros,
  gamma_init = nn_initializers_ones
)

stax_elementwise(fun, ...)

stax_Tanh

stax_Relu

stax_Exp

stax_LogSoftmax

stax_Softmax

stax_Softplus

stax_Sigmoid

stax_Elu

stax_LeakyRelu

stax_Selu

stax_Gelu

stax_MaxPool(window_shape, strides = NULL, padding = "VALID", spec = NULL)

stax_SumPool(window_shape, strides = NULL, padding = "VALID", spec = NULL)

stax_AvgPool(window_shape, strides = NULL, padding = "VALID", spec = NULL)

stax_Flatten

stax_Identity

stax_FanOut(num)

stax_FanInSum

stax_FanInConcat(axis = -1L)

stax_Dropout(rate, mode = "train")
```

## Arguments

- out_dim:

  Output dimension of `stax_Dense`.

- W_init, b_init:

  Initializers of the weights and biases (see
  [nn_initializers](https://adamrsophie.github.io/jaxR/reference/nn_initializers.md)).
  The default `W_init` of the convolutions is a Glorot normal
  initializer over the kernel's `I` and `O` axes.

- dimension_numbers:

  Three layout strings (input, kernel, output).

- out_chan:

  Number of output channels.

- filter_shape:

  Integer vector of kernel spatial sizes.

- strides:

  Integer vector of strides (default 1).

- padding:

  `"VALID"`, `"SAME"`, `"SAME_LOWER"` or a list of `c(low, high)` pairs.

- axis:

  0-based axes standardized by `stax_BatchNorm`; axis of
  `stax_FanInConcat`.

- epsilon:

  Variance correction of `stax_BatchNorm`.

- center, scale:

  Whether `stax_BatchNorm` has a bias (`beta`) and a scale (`gamma`).

- beta_init, gamma_init:

  Initializers of `beta` and `gamma`.

- fun:

  Elementwise function.

- ...:

  Further arguments passed to `fun` by `stax_elementwise`.

- window_shape:

  Integer vector of window sizes over the spatial dimensions.

- spec:

  Optional layout string such as `"NHWC"` naming the batch and channel
  dimensions of the pooling input.

- num:

  Number of copies made by `stax_FanOut`.

- rate:

  Keep probability of `stax_Dropout` (kept values are divided by
  `rate`).

- mode:

  `"train"` applies dropout; any other value returns the inputs.

## Value

A layer `list(init_fun, apply_fun)`.

## Details

`stax_Dense` is a fully connected layer `inputs %*% W + b`.
`stax_GeneralConv` and `stax_GeneralConvTranspose` are (transposed)
convolutions with layouts given as three strings such as
`c("NHWC", "HWIO", "NHWC")`; `stax_Conv`, `stax_ConvTranspose` and
`stax_Conv1DTranspose` fix the layouts to `NHWC`/`HWIO` and `NHC`/`HIO`.
`stax_BatchNorm` standardizes over `axis`. `stax_MaxPool`,
`stax_SumPool` and `stax_AvgPool` pool over windows of the spatial
dimensions (the dimensions other than `N` and `C` of `spec`, by default
the first and last). `stax_elementwise` makes a layer from an
elementwise function; `stax_Tanh`, `stax_Relu`, `stax_Exp`,
`stax_LogSoftmax`, `stax_Softmax`, `stax_Softplus`, `stax_Sigmoid`,
`stax_Elu`, `stax_LeakyRelu`, `stax_Selu`, `stax_Gelu`, `stax_Flatten`,
`stax_Identity` and `stax_FanInSum` are layers. `stax_FanOut` copies its
input `num` times, `stax_FanInSum` adds a list of inputs and
`stax_FanInConcat` concatenates them along `axis`.

## Examples

``` r
if (FALSE) { # \dontrun{
layer <- stax_serial(stax_Dense(4L), stax_Relu, stax_Dense(1L))
res <- layer$init_fun(random_key(0), c(-1L, 3L))
layer$apply_fun(res$params, jnp_ones(c(2, 3)))
} # }
```
