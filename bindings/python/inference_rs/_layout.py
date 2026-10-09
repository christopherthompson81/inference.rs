import ctypes
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from . import _native
from ._errors import check
from ._handle import Handle, Lease, bytes_arg, c_string_arg
from ._native import borrowed, lib

# Pass as a threshold to use the model's default (0.5).
DEFAULT_THRESHOLD = -1.0
BOX_FLOATS = 4


class PixelFormat(IntEnum):
    RGB8 = 0
    BGR8 = 1
    RGBA8 = 2
    BGRA8 = 3
    GRAY8 = 4


BYTES_PER_PIXEL = {
    PixelFormat.RGB8: 3,
    PixelFormat.BGR8: 3,
    PixelFormat.RGBA8: 4,
    PixelFormat.BGRA8: 4,
    PixelFormat.GRAY8: 1,
}


@dataclass(frozen=True)
class LayoutImage:
    """An 8-bit image, copied during detection; a stride of 0 means tightly packed rows."""

    pixels: bytes
    width: int
    height: int
    format: PixelFormat
    stride: int = 0

    def __post_init__(self):
        # The library cannot see the buffer's length, so a short one would be read past its end.
        row = self.width * BYTES_PER_PIXEL[PixelFormat(self.format)]
        stride = self.stride or row
        if stride < row:
            raise ValueError(f"stride {self.stride} is shorter than a row of {row} bytes")
        needed = stride * (self.height - 1) + row if self.height else 0
        if len(self.pixels) < needed:
            raise ValueError(f"{len(self.pixels)} pixel bytes; a {self.width}x{self.height} image needs {needed}")


@dataclass(frozen=True)
class LayoutDetection:
    """One region, in reading order; `box` is x1, y1, x2, y2 and `polygon` its outline's (x, y) vertices, in
    source-image pixels (the box's corners when the mask gives no outline)."""

    class_id: int
    label: str
    score: float
    box: tuple
    polygon: tuple


class LayoutModel:
    """A PP-DocLayoutV3 document layout detector, from an HF directory or a GGUF of it. Close it, or use `with`."""

    def __init__(self, path, backend: str | None = None, device: int = 0, threads: int = 0):
        config = _native.BackendConfig(c_string_arg(backend, "backend"), device, threads)
        model = ctypes.c_void_p()
        check(
            lib.inference_layout_model_load(
                c_string_arg(path, "path"),
                ctypes.byref(config),
                ctypes.byref(model),
            ),
            "inference_layout_model_load",
        )
        self._model = Handle(model.value, lib.inference_layout_model_free)
        self._finalizer = weakref.finalize(self, self._model.close)

    @property
    def labels(self) -> list[str]:
        labels = []
        with Lease(self._model) as model:
            for index in range(lib.inference_layout_model_label_count(model)):
                label = ctypes.c_void_p()
                check(
                    lib.inference_layout_model_label(model, index, ctypes.byref(label)),
                    "inference_layout_model_label",
                )
                labels.append(borrowed(label))
        return labels

    def detect(self, image: LayoutImage, threshold: float = DEFAULT_THRESHOLD) -> list[LayoutDetection]:
        return self.detect_batch([image], threshold)[0]

    def detect_batch(self, images: Sequence[LayoutImage], threshold: float = DEFAULT_THRESHOLD):
        """Detects on every image in one batched forward."""
        if not images:
            return []
        # The bytes objects stay alive here, so their buffers are passed without a copy.
        pixels = [bytes_arg(image.pixels) for image in images]
        native = (_native.Image * len(images))(
            *[
                _native.Image(
                    ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p),
                    image.width,
                    image.height,
                    image.stride,
                    image.format,
                )
                for image, data in zip(images, pixels)
            ]
        )
        results = (ctypes.c_void_p * len(images))()
        with Lease(self._model) as model:
            check(
                lib.inference_layout_detect_batch(
                    model,
                    native,
                    len(images),
                    threshold,
                    ctypes.cast(results, _native.out),
                ),
                "inference_layout_detect_batch",
            )
        try:
            return [self._read(result) for result in results]
        finally:
            for result in results:
                lib.inference_layout_result_free(result)

    @staticmethod
    def _read(result) -> list[LayoutDetection]:
        detections = []
        for index in range(lib.inference_layout_result_count(result)):
            class_id = ctypes.c_int32()
            label = ctypes.c_void_p()
            score = ctypes.c_float()
            box = (ctypes.c_float * BOX_FLOATS)()
            check(
                lib.inference_layout_result_detection(
                    result,
                    index,
                    ctypes.byref(class_id),
                    ctypes.byref(label),
                    ctypes.byref(score),
                    box,
                ),
                "inference_layout_result_detection",
            )
            points = ctypes.c_void_p()
            count = ctypes.c_size_t()
            check(
                lib.inference_layout_result_polygon(result, index, ctypes.byref(points), ctypes.byref(count)),
                "inference_layout_result_polygon",
            )
            flat = ctypes.cast(points, ctypes.POINTER(ctypes.c_float))[: 2 * count.value]
            polygon = tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))
            detections.append(LayoutDetection(class_id.value, borrowed(label), score.value, tuple(box), polygon))
        return detections

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
