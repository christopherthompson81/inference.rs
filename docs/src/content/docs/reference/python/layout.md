---
title: Layout
description: "PP-DocLayoutV3 document layout detection."
sidebar:
  order: 8
---
## `LayoutDetection`

One region, in reading order; `box` is x1, y1, x2, y2 in source-image pixels.

| Field | Type |
| --- | --- |
| `class_id` | `int` |
| `label` | `str` |
| `score` | `float` |
| `box` | `tuple` |


## `LayoutImage`

An 8-bit image, copied during detection; a stride of 0 means tightly packed rows.

| Field | Type | Default |
| --- | --- | --- |
| `pixels` | `bytes` | required |
| `width` | `int` | required |
| `height` | `int` | required |
| `format` | `PixelFormat` | required |
| `stride` | `int` | `0` |

### `LayoutImage.__post_init__`

```text
__post_init__()
```


## `LayoutModel`

A PP-DocLayoutV3 document layout detector. Close it, or use `with`.

### `LayoutModel.__init__`

```text
__init__(
    model_dir,
    backend: str | None = None,
    device: int = 0,
    threads: int = 0,
)
```

### `LayoutModel.labels`

```text
labels() -> list[str]
```

### `LayoutModel.detect`

```text
detect(
    image: LayoutImage,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[LayoutDetection]
```

### `LayoutModel.detect_batch`

```text
detect_batch(
    images: Sequence[LayoutImage],
    threshold: float = DEFAULT_THRESHOLD,
)
```

Detects on every image in one batched forward.

### `LayoutModel._read`

```text
_read(result) -> list[LayoutDetection]
```

### `LayoutModel.close`

```text
close()
```

### `LayoutModel.__enter__`

```text
__enter__()
```

### `LayoutModel.__exit__`

```text
__exit__()
```


## `PixelFormat`

---

<small>Generated from [`bindings/python/inference_rs`](https://github.com/christopherthompson81/inference.rs/blob/master/bindings/python/inference_rs).</small>
