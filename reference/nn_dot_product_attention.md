# Scaled dot product attention

Port of `jax.nn.dot_product_attention` (the `"xla"` implementation): for
each head, `softmax(Q K^T * scale + bias) V`. Shapes use `B` batch, `T`
query length, `S` key/value length, `N` query heads, `K` key/value heads
(`N` must be a multiple of `K`: grouped and multi-query attention) and
`H` head dimension.

## Usage

``` r
nn_dot_product_attention(
  query,
  key,
  value,
  bias = NULL,
  mask = NULL,
  scale = NULL,
  is_causal = FALSE,
  query_seq_lengths = NULL,
  key_value_seq_lengths = NULL,
  local_window_size = NULL,
  implementation = NULL,
  return_residual = FALSE
)
```

## Arguments

- query:

  Query array of shape `(B, T, N, H)` or `(T, N, H)`.

- key:

  Key array of shape `(B, S, K, H)` or `(S, K, H)`.

- value:

  Value array with the shape of `key`.

- bias:

  Optional bias added to the logits, broadcastable to `(B, N, T, S)`.

- mask:

  Optional boolean mask broadcastable to `(B, N, T, S)`; `TRUE` marks
  the elements that take part in attention.

- scale:

  Scale of the logits (default `1 / sqrt(H)`).

- is_causal:

  Apply a causal mask.

- query_seq_lengths, key_value_seq_lengths:

  Optional `int32` arrays of shape `(B)` with the valid lengths of the
  query and key/value sequences.

- local_window_size:

  Optional `c(left, right)` window (a single value is used for both
  sides): each query attends to keys at offsets `-left` to `right`.

- implementation:

  `NULL` or `"xla"` (`"cudnn"` needs a GPU and is not available).

- return_residual:

  Also return the log-sum-exp of the logits, of shape `(B, T, N)` or
  `(T, N)`.

## Value

An array with the shape of `query`, or `list(output, residual)`.

## Examples

``` r
if (FALSE) { # \dontrun{
q <- random_normal(random_key(0), c(2, 3, 4, 5))
k <- random_normal(random_key(1), c(2, 6, 2, 5))
nn_dot_product_attention(q, k, k, is_causal = TRUE)
} # }
```
