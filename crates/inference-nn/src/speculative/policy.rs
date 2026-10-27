/// Depths a built-in MTP head on automatic depth chooses between; the last is its maximum. One draft stays in:
/// it wins when verification is expensive, as with MoE experts offloaded to the CPU.
pub const AUTO_DEPTHS: [usize; 4] = [1, 2, 3, 4];
pub const AUTO_MAX_DEPTH: usize = AUTO_DEPTHS[AUTO_DEPTHS.len() - 1];
// Depths other than a drafter's primary one are graphed up to this batch; larger batches run them eagerly
const AUTO_EXTRA_DEPTH_GRAPH_MAX_BATCH: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchPlan {
    pub proposal_len: usize,
    pub needs_target_hiddens: bool,
}

impl SpeculativeBatchPlan {
    pub const fn new(proposal_len: usize) -> Self {
        Self {
            proposal_len,
            needs_target_hiddens: true,
        }
    }

    pub const fn without_target_hiddens(mut self) -> Self {
        self.needs_target_hiddens = false;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeGraphPlan {
    pub proposal_len: usize,
    pub max_batch_size: Option<usize>,
}

impl SpeculativeGraphPlan {
    pub const fn new(proposal_len: usize, max_batch_size: Option<usize>) -> Self {
        Self {
            proposal_len,
            max_batch_size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeculativeBatchObservation {
    pub batch_size: usize,
    pub proposal_len: usize,
    pub sequences: usize,
    pub proposed_drafts: usize,
    pub accepted_drafts: usize,
}

/// `candidates` below `max_depth`, plus `max_depth` itself.
pub fn depths_up_to(candidates: &[usize], max_depth: usize) -> Vec<usize> {
    let mut depths = candidates
        .iter()
        .copied()
        .filter(|depth| *depth < max_depth)
        .collect::<Vec<_>>();
    depths.push(max_depth);
    depths
}

/// Graph plans for autotuned `depths`: `primary` at every batch size, the others for small batches.
pub fn auto_depth_graph_plans(depths: &[usize], primary: usize) -> Vec<SpeculativeGraphPlan> {
    depths
        .iter()
        .map(|&depth| {
            SpeculativeGraphPlan::new(
                depth,
                (depth != primary).then_some(AUTO_EXTRA_DEPTH_GRAPH_MAX_BATCH),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::SpeculativeBatchPlan;

    #[test]
    fn target_hiddens_are_required_by_default() {
        assert!(SpeculativeBatchPlan::new(7).needs_target_hiddens);
        assert!(
            !SpeculativeBatchPlan::new(7)
                .without_target_hiddens()
                .needs_target_hiddens
        );
    }
}
