# Backend, devices and processes

Ports of `jax.default_backend`, `jax.device_count`,
`jax.local_device_count`, `jax.local_devices`, `jax.process_index`,
`jax.process_count` and `jax.process_indices` (and the deprecated
aliases `jax.host_id`, `jax.host_count`, `jax.host_ids`, which warn as
in JAX). jaxR runs one process on the devices of one PJRT client.

## Usage

``` r
default_backend()

device_count(backend = NULL)

local_device_count(backend = NULL)

local_devices(process_index = NULL, backend = NULL, host_id = NULL)

process_index(backend = NULL)

process_count(backend = NULL)

process_indices(backend = NULL)

host_id(backend = NULL)

host_count(backend = NULL)

host_ids(backend = NULL)
```

## Arguments

- backend:

  Ignored (one backend).

- process_index:

  Process index (only 0 exists).

- host_id:

  Deprecated alias of `process_index`.

## Value

The platform name, a count, an index, or device descriptions (as
[`devices()`](https://adamrsophie.github.io/jaxR/reference/devices.md)).
