# Model inventory investigation

Building engine-free inventory calls for the C ABI (inspect a model, scan folders and the Hugging Face cache, delete
from the cache, download with progress) for a model manager in the Avalonia demo. The runs below are against a real
local model folder (a few hundred GB of GGUF, safetensors and `.nemo` models from several projects) and a real HF
cache (109 repositories).

## Run 1 - 2026-10-12 19:10

Question: does a first-cut scan classify a real folder correctly, and how fast is it? A throwaway test calling
`scan_models` on the folder plus the cache, printing each entry.

182 entries in 25.7 s. Classification problems:
- every `.nemo` failed with "Not a directory (os error 20)": detection read `<file>/config.json`, which errors with
  ENOTDIR rather than NotFound; the same error sent ASR and Kokoro GGUFs to the "text" fallback
- other projects' GGUFs (`general.architecture = audiocpp`, `bert`) were labeled text, though nothing here loads them
- a `*-imatrix.gguf` (llama.cpp importance matrix) showed as a broken model
- cache entries holding only a probed `config.json` were listed (and one misrouted, without its weight index)
- unsupported architectures reported "Unsupported Hugging Face Transformers -CausalLM model class " with no name

Fixes: detection treats NotADirectory as no config; `inference_gguf::loads_architecture` (the canonical text
registry, the multimodal families' text architectures, the Qwen multimodal architectures) gates the text fallback,
anything else reports "unsupported GGUF architecture"; imatrix GGUFs are skipped by `general.type = imatrix` (not by
name: `Model-IMatrix-Q5_K_M.gguf` is a model quantized with one, and a selection test pins that); directories need a
weight file; the error names the architecture.

## Run 2 - 2026-10-12 19:25

Question: where do the 25 s go? Per-directory timings, then per-call. The cache costs 37 ms; the local folder 24.6 s,
16 s of it one 890 MB Kokoro GGUF whose voices and phonemizer data live in metadata arrays. `GgufArchive::open`
builds a `Value` per array element, so it reads all 890 MB (16 s at disk speed).

Fix: `gguf_file::Content::read` split into `Header::read` (the same parse, tensor dtypes left as raw GGML ids so
files this crate cannot run still inspect) plus the dtype conversion. Inventory and HiDream's `is_gguf` probe use
the header. Scan: 28 s -> 7.5 s.

## Run 3 - 2026-10-12 19:40

Question: what is left in 7.5 s (warm cache, so CPU)? Each GGUF costs 70-113 ms twice: inventory's header read
and detection's HiDream probe, both parsing the tokenizer's 150k-248k strings.

Fixes: `Header::read_summary` skips arrays longer than 64 values (strings read through the reader's buffer: a
position query per string cost a syscall each, which kept the first version at 70 ms); inventory only runs the
special-model probes for architectures that are not loadable text GGUFs (a first version built the `(text,
detect(..))` tuple eagerly and still ran them). Header 70 ms -> 9 ms per file; scan 7.5 s -> 0.57 s, same 145
models and kinds.

## Run 4 - 2026-10-12 20:30

Question: does the download stream work against the Hub, and does cancel stop it? Python bindings, a throwaway
`HF_HUB_CACHE`: a 659-byte `config.json`, then a 988 MB `model.safetensors` canceled after its third progress event.

The small file reported 100% twice (the per-percent update and an explicit final one). The cancel ended the stream
with `canceled`, but ~100 more progress events followed and the whole file landed: hf-hub's async downloader spawns
each chunk as a task, and dropping the download future leaves them running.

Fixes: the final report only when the per-percent updates did not reach 100% (an empty file); each download runs on
its own small runtime on a dedicated thread, shut down once the work ends, which ends any chunk tasks still running
(the partial file keeps hf-hub's committed-bytes footer, so it resumes later). After: canceled at 20 MB, the cache
still 20 MB five seconds later; the event stream drops the download when it is dropped too.
