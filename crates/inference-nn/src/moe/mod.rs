mod experts;
mod router;

use inference_quant::Shard;

pub use experts::{ExpertProj, ExpertProjNames, MoEExperts, MoEExpertsConfig, prelog_moe_backend};
pub use experts::{expert_stack_available, rebuild_expert_projection};
pub use router::{GroupedRouter, GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
