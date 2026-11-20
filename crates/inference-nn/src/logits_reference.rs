//! Scoring logits against a reference distribution in llama-perplexity's KL-divergence base encoding, and reading
//! such a base file (its tokens, chunk layout and per-row records).

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use inference_tensor::{DType, Device, Result, Tensor};

// reference log-probs at or below this add nothing to the divergence, as llama-perplexity sums it
const REFERENCE_LOG_PROB_FLOOR: f64 = -16.0;
// rows of the divergence computed at once, which bounds its temporaries (a few `[rows, vocab]` f32 tensors)
const DIVERGENCE_ROWS: usize = 64;
const KLD_BASE_MAGIC: &[u8; 8] = b"_logits_";

/// Reference log-probabilities for consecutive logits rows from `first_row`, in llama-perplexity's KL-divergence base
/// encoding: row `r`'s token `t` has `log p = scales[r] * codes[r * width + t] + floors[r]`.
#[derive(Debug, Clone, Default)]
pub struct LogitsReference {
    pub first_row: usize,
    /// Codes per row, at least the model's vocabulary (the ones past it are ignored).
    pub width: usize,
    pub scales: Vec<f32>,
    pub floors: Vec<f32>,
    /// `scales.len() * width` little-endian u16 codes.
    pub codes: Vec<u8>,
}

/// Per reference row: `KL(reference || model)`, whether both put the same token on top, and the reference log-prob of
/// the prompt's next token.
#[derive(Debug, Clone, Default)]
pub struct Divergence {
    pub kld: Vec<f32>,
    pub same_top: Vec<bool>,
    pub reference_logprobs: Vec<f32>,
}

// each row's first index of its maximum, as llama-perplexity picks the top token (a device argmax may take any tie)
fn first_max(rows: &Tensor, positions: &Tensor) -> inference_tensor::Result<Tensor> {
    let at_max = rows.broadcast_eq(&rows.max_keepdim(1)?)?;
    let width = u32::try_from(positions.dim(1)?).map_err(inference_tensor::Error::wrap)?;
    let past = positions.affine(0.0, f64::from(width))?;
    at_max
        .where_cond(
            &positions.broadcast_as(rows.shape())?,
            &past.broadcast_as(rows.shape())?,
        )?
        .min(1)
}

/// A reference's base log-probabilities decoded on a device, for scoring any number of logits against it.
pub struct DecodedReference {
    first_row: usize,
    blocks: Vec<DecodedBlock>,
    positions: Tensor,
    reference_logprobs: Vec<f32>,
}

// up to DIVERGENCE_ROWS rows of a decoded reference: their log-probabilities, which of them count, and their top
struct DecodedBlock {
    start: usize,
    len: usize,
    base: Tensor,
    counted: Tensor,
    top: Tensor,
}

/// `reference` decoded on `device` for `vocab`-wide logits over `tokens`.
pub fn decode_reference(
    reference: &LogitsReference,
    tokens: &[u32],
    vocab: usize,
    device: &Device,
) -> inference_tensor::Result<DecodedReference> {
    let n = reference.scales.len();
    let width = reference.width;
    if reference.floors.len() != n || reference.codes.len() != 2 * n * width || width < vocab {
        inference_tensor::bail!(
            "a reference of {n} rows of {width} does not fit a vocab of {vocab}"
        );
    }
    let vocab_u32 = u32::try_from(vocab).map_err(inference_tensor::Error::wrap)?;
    let positions = Tensor::arange(0, vocab_u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?;
    let mut blocks = Vec::new();
    let mut reference_logprobs = Vec::with_capacity(n);
    for start in (0..n).step_by(DIVERGENCE_ROWS) {
        let len = DIVERGENCE_ROWS.min(n - start);
        let bytes = &reference.codes[2 * start * width..][..2 * len * width];
        let pairs = Tensor::from_slice(bytes, (len, width, 2), device)?.narrow(1, 0, vocab)?;
        let codes = pairs
            .narrow(2, 0, 1)?
            .to_dtype(DType::F32)?
            .add(
                &pairs
                    .narrow(2, 1, 1)?
                    .to_dtype(DType::F32)?
                    .affine(256.0, 0.0)?,
            )?
            .squeeze(2)?;
        let column = |values: &[f32]| Tensor::from_slice(&values[start..][..len], (len, 1), device);
        let base = codes
            .broadcast_mul(&column(&reference.scales)?)?
            .broadcast_add(&column(&reference.floors)?)?;
        let counted = base.gt(REFERENCE_LOG_PROB_FLOOR)?.to_dtype(DType::F32)?;
        let top = first_max(&base, &positions)?;
        let next = tokens[reference.first_row + start + 1..][..len]
            .iter()
            .map(|&token| token.min(vocab_u32 - 1))
            .collect::<Vec<_>>();
        let next = Tensor::from_slice(&next, (len, 1), device)?;
        reference_logprobs.extend(base.gather(&next, 1)?.flatten_all()?.to_vec1::<f32>()?);
        blocks.push(DecodedBlock {
            start,
            len,
            base,
            counted,
            top,
        });
    }
    Ok(DecodedReference {
        first_row: reference.first_row,
        blocks,
        positions,
        reference_logprobs,
    })
}

/// Per row from the reference's first row, the KL divergence of `logits` `(rows, vocab)` from the reference, our top
/// token's agreement with its top, and the reference's log-probability of the next token.
pub fn divergence(
    logits: &Tensor,
    tokens: &[u32],
    reference: &LogitsReference,
) -> inference_tensor::Result<Divergence> {
    let (rows, vocab) = logits.dims2()?;
    if reference.first_row + reference.scales.len() >= rows {
        inference_tensor::bail!(
            "a reference of {} rows from row {} does not fit {rows} rows",
            reference.scales.len(),
            reference.first_row
        );
    }
    divergence_decoded(
        logits,
        &decode_reference(reference, tokens, vocab, logits.device())?,
    )
}

/// As [`divergence`], against a reference decoded once.
pub fn divergence_decoded(
    logits: &Tensor,
    reference: &DecodedReference,
) -> inference_tensor::Result<Divergence> {
    let mut out = Divergence {
        reference_logprobs: reference.reference_logprobs.clone(),
        ..Default::default()
    };
    for block in &reference.blocks {
        let ours = inference_tensor::nn::ops::log_softmax(
            &logits.narrow(0, reference.first_row + block.start, block.len)?,
            1,
        )?;
        let kld = block
            .base
            .exp()?
            .mul(&block.base.sub(&ours)?)?
            .mul(&block.counted)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let same = first_max(&ours, &reference.positions)?
            .eq(&block.top)?
            .to_vec1::<u8>()?;
        out.kld.extend(kld);
        out.same_top.extend(same.into_iter().map(|same| same != 0));
    }
    Ok(out)
}

/// A llama-perplexity KL-divergence base: `n_chunk` chunks of `n_ctx` tokens, each chunk's second half recorded as
/// rows of `n_vocab` codes (see [`LogitsReference`]).
pub struct KldBase {
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub n_chunk: usize,
    pub tokens: Vec<u32>,
    reader: BufReader<File>,
}

impl KldBase {
    /// The base at `path`, its header and tokens read; the records follow chunk by chunk through [`Self::next_chunk`].
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != KLD_BASE_MAGIC {
            inference_tensor::bail!(
                "{} is not a llama-perplexity KL divergence base",
                path.display()
            );
        }
        let mut int = || -> Result<usize> {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            usize::try_from(i32::from_le_bytes(b)).map_err(inference_tensor::Error::wrap)
        };
        let (n_ctx, n_vocab, n_chunk) = (int()?, int()?, int()?);
        let mut bytes = vec![0u8; 4 * n_ctx * n_chunk];
        reader.read_exact(&mut bytes)?;
        let tokens = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::try_from(i32::from_le_bytes(*b)).map_err(inference_tensor::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            n_ctx,
            n_vocab,
            n_chunk,
            tokens,
            reader,
        })
    }

    /// Whether `path` starts as a KL-divergence base does.
    pub fn is_base(path: &Path) -> bool {
        let mut magic = [0u8; 8];
        File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .is_ok_and(|()| &magic == KLD_BASE_MAGIC)
    }

    /// The first scored row of a chunk (llama-perplexity scores the second half, each row predicting the next token).
    pub fn first_row(&self) -> usize {
        self.n_ctx / 2
    }

    /// The next chunk's records as a reference from [`Self::first_row`].
    pub fn next_chunk(&mut self) -> Result<LogitsReference> {
        let first = self.first_row();
        let rows = self.n_ctx - 1 - first;
        let width = 2 * self.n_vocab.div_ceil(2);
        let record = 2 * (width + 4);
        let mut bytes = vec![0u8; record * rows];
        self.reader.read_exact(&mut bytes)?;
        let mut reference = LogitsReference {
            first_row: first,
            width,
            codes: Vec::with_capacity(rows * 2 * width),
            ..Default::default()
        };
        for row in bytes.chunks_exact(record) {
            let float =
                |at: usize| f32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]);
            reference.scales.push(float(0));
            reference.floors.push(float(4));
            reference.codes.extend_from_slice(&row[8..]);
        }
        Ok(reference)
    }
}
