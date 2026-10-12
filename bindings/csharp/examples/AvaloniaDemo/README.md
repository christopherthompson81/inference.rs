# Avalonia demo

A desktop app (Avalonia 12, Linux, Windows and macOS) that runs the engine through the C# bindings, a tab per modality:

- **Chat:** streaming replies with Stop, sampling settings, the model's reasoning, and images or audio attached to a
  message, from a Hugging Face id, a model directory, or a GGUF with its mmproj.
- **Speech to text:** an audio file's transcript with segment timestamps, and speaker labels when a diarization model
  is loaded beside the recognizer.
- **Text to speech:** a WAV from Kokoro or Dia, played with the system player or saved.
- **Images:** text to image with HiDream-O1 or FLUX, and HiDream-O1's edits: one reference image to edit, or several
  as subjects for a new scene.
- **Documents:** PP-DocLayoutV3 layout regions drawn over the page, each read by an OCR model such as PaddleOCR-VL with
  the prompt for its kind (text, table, formula, chart).
- **Engine:** the chat engine's models with unload and reload, its cache and speculative-decoding metrics, and the
  system report.

Each tab loads its own models, showing the load's progress (downloads included) and the C ABI's errors. Chat models
take a context length, which sizes their KV cache so several models fit on one GPU.

```bash
cargo build --release -p inference-ffi --features cuda   # or without cuda; the bindings find target/release
dotnet run --project bindings/csharp/examples/AvaloniaDemo
```

It depends only on the bindings project. `scripts/local_ci.sh --bindings` builds it with the solution.
