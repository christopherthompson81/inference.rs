use inference_tensor::{Result, Tensor};

use crate::model::{IsqModel, ModelForwardContext, NormalModel};

/// The text model a LLaVA wrapper runs its merged image and text embeddings through.
pub trait LLaVALLM: IsqModel + NormalModel + Sync + Send {
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor>;
    fn forward_input_embed(
        &self,
        input_ids: &Tensor,  // only for masking
        input_embed: Tensor, // we don't want to clone, so we pass it in
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor>;
}

impl LLaVALLM for Llama {
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.get_input_embeddings(input_ids)
    }
    fn forward_input_embed(
        &self,
        input_ids: &Tensor,
        input_embed: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(input_ids, input_embed, ctx)
    }
}

impl LLaVALLM for Mistral {
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.get_input_embeddings(input_ids)
    }
    fn forward_input_embed(
        &self,
        input_ids: &Tensor,
        input_embed: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_embeds(input_ids, input_embed, ctx)
    }
}

pub use crate::llama::Llama;
pub use crate::mistral::Model as Mistral;
