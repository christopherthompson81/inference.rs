//! What an embedding model takes beside its token ids: a pair's segment ids and each row's task adapter.

use inference_tensor::{D, DType, Result, Tensor};

const JINA_V5_MODEL_TYPE: &str = "jina_embeddings_v5";
// Jina v5's retrieval adapter serves both sides of a search, told apart by their prefix
const JINA_V5_RETRIEVAL: &str = "retrieval";
const JINA_V5_QUERY: (&str, &str) = ("retrieval.query", "Query: ");
const JINA_V5_PASSAGE: (&str, &str) = ("retrieval.passage", "Document: ");
const JINA_V5_DOCUMENT_PREFIX: &str = "Document: ";

/// The tasks a model's adapters serve, with the instruction each prepends to a text prompt.
#[derive(Clone, Debug, Default)]
pub struct EmbeddingTasks {
    pub names: Vec<String>,
    pub instructions: Vec<String>,
    /// Each task's adapter, by the name it ships under in `adapters/`; empty when they are in the model's weights.
    pub adapters: Vec<String>,
}

impl EmbeddingTasks {
    pub fn index(&self, task: &str) -> Option<usize> {
        self.names.iter().position(|name| name == task)
    }
}

#[derive(inference_json::FromJson)]
struct JinaV5Config {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    task_names: Vec<String>,
}

/// Jina v5's tasks, under the names Jina's API takes: one per shipped adapter, retrieval split by side.
pub fn jina_v5_tasks(config: &str) -> serde_json::Result<Option<EmbeddingTasks>> {
    let config: JinaV5Config = serde_json::from_str(config)?;
    if config.model_type != JINA_V5_MODEL_TYPE {
        return Ok(None);
    }
    let mut tasks = EmbeddingTasks::default();
    for adapter in config.task_names {
        let sides = if adapter == JINA_V5_RETRIEVAL {
            vec![JINA_V5_QUERY, JINA_V5_PASSAGE]
        } else {
            vec![(adapter.as_str(), JINA_V5_DOCUMENT_PREFIX)]
        };
        for (name, prefix) in sides {
            tasks.names.push(name.to_string());
            tasks.instructions.push(prefix.to_string());
            tasks.adapters.push(adapter.clone());
        }
    }
    Ok(Some(tasks))
}

/// Each row's task: `ids` (u32, `[batch]`) picks an adapter, `active` (f32, `[batch]`) is 0 for a row with none.
#[derive(Clone)]
pub struct TaskRows {
    pub ids: Tensor,
    pub active: Tensor,
}

impl TaskRows {
    /// `[batch, 1, 1]` in `dtype`: a row's scale on its adapter's delta.
    fn scale(&self, scaling: f64, dtype: DType) -> Result<Tensor> {
        let batch = self.active.dim(0)?;
        (&self.active * scaling)?
            .reshape((batch, 1, 1))?
            .to_dtype(dtype)
    }
}

/// The per-row inputs of an embedding forward.
#[derive(Clone, Default)]
pub struct EmbeddingExtras {
    /// The segment ids of a text pair; models without segment embeddings ignore them.
    pub token_type_ids: Option<Tensor>,
    pub tasks: Option<TaskRows>,
    /// A decision model's question type and option markers per row.
    pub decision: Option<crate::decision::DecisionRows>,
}

/// A linear layer's adapters, one per task, stacked: `a` is `[tasks, rank, in]`, `b` `[tasks, out, rank]`.
pub struct TaskLora {
    a: Tensor,
    b: Tensor,
    scaling: f64,
}

impl TaskLora {
    pub fn new(a: Tensor, b: Tensor, scaling: f64) -> Self {
        Self { a, b, scaling }
    }

    /// The low-rank delta `scaling * x A_t^T B_t^T` of each row's task, zero for a row without one.
    pub fn delta(&self, xs: &Tensor, rows: &TaskRows) -> Result<Tensor> {
        let a = self.a.index_select(&rows.ids, 0)?.to_dtype(xs.dtype())?;
        let b = self.b.index_select(&rows.ids, 0)?.to_dtype(xs.dtype())?;
        xs.matmul(&a.transpose(1, 2)?)?
            .matmul(&b.transpose(1, 2)?)?
            .broadcast_mul(&rows.scale(self.scaling, xs.dtype())?)
    }

    pub fn a(&self) -> &Tensor {
        &self.a
    }

    pub fn b(&self) -> &Tensor {
        &self.b
    }
}

/// An embedding table's adapters, one per task: `a` is `[tasks, vocab, rank]`, `b` `[tasks, rank, hidden]`.
pub struct TaskEmbeddingLora {
    a: Tensor,
    b: Tensor,
    scaling: f64,
}

impl TaskEmbeddingLora {
    pub fn new(a: Tensor, b: Tensor, scaling: f64) -> Self {
        Self { a, b, scaling }
    }

    /// `scaling * A_t[ids] B_t` for each row's task: the rows of `ids` (`[batch, seq]`) in every task's `a`, then
    /// the row's own task picked out.
    pub fn delta(&self, ids: &Tensor, rows: &TaskRows, dtype: DType) -> Result<Tensor> {
        let (tasks, vocab, rank) = self.a.dims3()?;
        let (batch, seq_len) = ids.dims2()?;
        // vocab rows hold every task's rank-wide slice side by side
        let table = self.a.transpose(0, 1)?.reshape((vocab, tasks * rank))?;
        let picked = table
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((batch, seq_len, tasks, rank))?;
        let index = rows
            .ids
            .reshape((batch, 1, 1, 1))?
            .broadcast_as((batch, seq_len, 1, rank))?
            .contiguous()?;
        let a_rows = picked.gather(&index, D::Minus2)?.squeeze(D::Minus2)?;
        let b = self.b.index_select(&rows.ids, 0)?;
        a_rows
            .to_dtype(dtype)?
            .matmul(&b.to_dtype(dtype)?)?
            .broadcast_mul(&rows.scale(self.scaling, dtype)?)
    }

    pub fn a(&self) -> &Tensor {
        &self.a
    }

    pub fn b(&self) -> &Tensor {
        &self.b
    }
}
