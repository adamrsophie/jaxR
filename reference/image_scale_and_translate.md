# Scale and translate an image

Port of `jax.image.scale_and_translate`: resamples `image` to `shape` so
that a location `x` of the input maps to `x * scale + translation` along
each dimension in `spatial_dims` (pixels are centered at half integers).
Output locations whose source lies outside the input are zero.

## Usage

``` r
image_scale_and_translate(
  image,
  shape,
  spatial_dims,
  scale,
  translation,
  method,
  antialias = TRUE,
  precision = "highest"
)
```

## Arguments

- image:

  An array.

- shape:

  Integer vector, the output shape.

- spatial_dims:

  0-based dimensions to transform.

- scale, translation:

  Arrays (or numbers) with one entry per spatial dimension.

- method:

  A method name (see
  [image_ResizeMethod](https://adamrsophie.github.io/jaxR/reference/image_ResizeMethod.md));
  `"nearest"` is not supported.

- antialias:

  Antialias when downsampling.

- precision:

  Precision of the contractions.

## Value

The resampled array.
