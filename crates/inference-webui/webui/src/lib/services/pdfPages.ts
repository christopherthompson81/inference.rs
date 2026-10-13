import type { PDFDocumentProxy } from "pdfjs-dist";

// pages render at this resolution: a Letter page comes out about 1700x2200, enough for OCR of body text
const RENDER_DPI = 200;
const PDF_POINTS_PER_INCH = 72;

export interface RenderedPage {
  blob: Blob;
  width: number;
  height: number;
}

/** A PDF opened for rendering its pages; pdf.js loads only when a PDF is opened. */
export async function openPdf(file: Blob): Promise<PDFDocumentProxy> {
  // the legacy build polyfills what the modern one assumes (Uint8Array.toHex), which browsers a year old lack
  const [pdfjs, worker] = await Promise.all([
    import("pdfjs-dist/legacy/build/pdf.mjs"),
    import("pdfjs-dist/legacy/build/pdf.worker.min.mjs?url"),
  ]);
  pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
  return pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
}

/** Page `number` (from 1) as a PNG on white, at the render resolution. */
export async function renderPage(pdf: PDFDocumentProxy, number: number): Promise<RenderedPage> {
  const page = await pdf.getPage(number);
  const viewport = page.getViewport({ scale: RENDER_DPI / PDF_POINTS_PER_INCH });
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(viewport.width);
  canvas.height = Math.ceil(viewport.height);
  const context = canvas.getContext("2d")!;
  context.fillStyle = "white";
  context.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvas, canvasContext: context, viewport }).promise;
  page.cleanup();
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((png) => (png ? resolve(png) : reject(new Error("the page could not be drawn"))), "image/png"),
  );
  return { blob, width: canvas.width, height: canvas.height };
}
