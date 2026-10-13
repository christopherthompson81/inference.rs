# OCR batching investigation

The web UI's Documents tab reads a page's layout regions one request at a time. Does sending them concurrently, so
the engine batches them, raise throughput?

## Run 1 - 2026-10-13 15:40

**Question:** regions per second for one page at 1 to 16 OCR requests in flight.

**Setup:** a synthetic 1700x2200 page (a title and 8 sections of invented two-column text, drawn with PIL) laid out by
PP-DocLayoutV3 into 25 regions (`paragraph_title`, `text`); each region cropped and sent as
`/v1/chat/completions` (PaddleOCR-VL-1.5, prompt `OCR:`, temperature 0, 8192-token paged cache) from a thread pool,
after one warm-up request. Scratchpad `batch/bench.py`.

**Raw finding, OCR model loaded with `max_seqs: 16`:**

| in flight | time | tokens/s | regions/s | text matching 1 in flight |
|---|---|---|---|---|
| 1 | 3.6 s | 197 | 7.0 | |
| 2 | 1.6 s | 430 | 15.3 | 24/25 |
| 4 | 1.2 s | 591 | 21.1 | 25/25 |
| 8 | 1.1 s | 633 | 22.5 | 24/25 |
| 16 | 0.8 s | 858 | 30.5 | 24/25 |

The one region that differed read the same words; batched, it kept the region's line breaks where the serial read
joined its lines with spaces (BF16 batched kernels reduce in a different order).

**Raw finding, OCR model at the default `max_seqs` (4):** 1 in flight 3.6 s (6.9 regions/s); 4 in flight 1.2 s
(20.5 regions/s); 8 in flight 1.2 s (21.3 regions/s); text 25/25 the same. 4.4 GB of VRAM with both models loaded.

**Implication:** the engine batches concurrent reads: 3x at the default 4 sequences, 4.3x at 16. Requests past the
model's sequence limit wait in its queue at no cost, so the UI can keep up to 16 in flight whatever the model was
loaded with, and a model loaded with more sequences (the Sequences field) goes faster still.

## Run 2 - 2026-10-13 16:05

**Change:** the Documents view reads a page's regions with 16 requests in flight (each result lands on its own region,
so the order holds; one failure stops the rest taking regions), and swaps pages only once the next page is rendered
and decoded, rendering the page after it meanwhile.

**Raw finding:** a 3-page PDF of Run 1's page at the default 4 OCR sequences: Process all pages 5.5 s, layout
detection and saving the job included, against about 11 s at Run 1's serial 3.6 s a page for the reads alone (an
estimate, not a timed run of the old view). Over two Next and one Previous, and the whole run, no frame showed the
"Rendering the page" placeholder and the page's SVG was never replaced.

**Implication:** reading is 3x faster at the default sequences, and the page no longer flashes or reflows on a switch.
