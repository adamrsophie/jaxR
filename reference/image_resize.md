# Resize an image

Port of `jax.image.resize`. `shape` gives the size of every dimension
(batch and channel dimensions keep their size). Methods: see
[image_ResizeMethod](https://adamrsophie.github.io/jaxR/reference/image_ResizeMethod.md).
Except for `"nearest"` (which keeps the dtype), the result is floating
point; with `antialias` a downsampling kernel is widened to low-pass
filter the image.

## Usage

``` r
image_resize(image, shape, method, antialias = TRUE, precision = "highest")
```

## Arguments

- image:

  An array.

- shape:

  Integer vector, the output shape.

- method:

  A method name (see
  [image_ResizeMethod](https://adamrsophie.github.io/jaxR/reference/image_ResizeMethod.md)).

- antialias:

  Antialias when downsampling.

- precision:

  Precision of the contractions (`"highest"` as in JAX).

## Value

The resized array.

## Examples

``` r
if (FALSE) { # \dontrun{
img <- jnp_reshape(jnp_arange(16, dtype = "float32"), c(4L, 4L))
image_resize(img, c(2L, 2L), "linear")
} # }
```
