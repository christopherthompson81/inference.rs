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

## Run 3 - 2026-10-13 17:10

**Question:** with the regions read 16 at a time, does the batch drain between pages, and does running the layout
ahead keep it full?

**Change:** Process all pages runs as a pipeline: one task renders and lays out up to 2 pages ahead and queues their
regions; 16 readers take regions from any page. A page is kept when its last region is read; a failed page is marked
and the rest go on; saves run one after another.

**Raw finding:** same server and PDFs (Run 2's page, 3 and 6 pages, default 4 OCR sequences), the page reloaded
before each run, no other GPU work. A first run of each build is slower (4.5 to 5.0 s for 3 pages, warming caches).

| build | 3 pages | 6 pages | per page |
|---|---|---|---|
| a batch per page (master) | 4.0, 4.0 s | 8.1, 8.1 s | 1.33 s |
| pipelined | 3.5, 3.5 s | 7.2, 7.2 s | 1.18 s |
| pipelined, Chromium 153 and pdf.js's modern build | 3.7, 3.7 s | 7.1, 7.2 s | 1.19 s |

An earlier pipelined run took 6.5 s for the 3 pages while the LAN server on port 1234 shared the GPU; it was stopped
for these runs.

**Implication:** the run-ahead saves about 0.15 s a page, the gap while one page's last regions drained before the
next was laid out: about 11%. Kept. The tests now use Playwright's bundled Chromium (153) rather than the system's
131, which lacks `Uint8Array.toHex`; pdf.js's modern build works in it, so the legacy build is no longer needed.
