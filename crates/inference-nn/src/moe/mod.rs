mod experts;
mod router;

use inference_quant::Shard;

pub use experts::{ExpertProj, ExpertProjNames, MoEExperts, MoEExpertsConfig, prelog_moe_backend};
pub use experts::{OFFLOAD_STAGING_EXPERTS, expert_stack_available, rebuild_expert_projection};
#[cfg(feature = "cuda")]
pub use experts::{OFFLOADED_GRAPH_MAX_TOKENS, offload_activity};
pub use router::{GroupedRouter, GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
