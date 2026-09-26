mod experts;

use inference_quant::Shard;

pub use experts::{expert_stack_available, rebuild_expert_projection};
pub use experts::{prelog_moe_backend, ExpertProj, ExpertProjNames, MoEExperts, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
