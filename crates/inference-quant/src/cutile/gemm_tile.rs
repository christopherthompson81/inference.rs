//! The tile config and tuning space the FP8 GEMMs share.

use cutile::tile_kernel::CompileOptions;

use super::tune::{Space, config};

/// Launch config: the row tile, the swizzle map over output tiles, persistent tile blocks per SM,
/// and the knobs the autotuner sweeps. The column tile is pinned to one weight-scale column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct GemmTileConfig {
    pub bm: i32,
    pub map_m: i32,
    pub map_n: i32,
    pub blocks_per_sm: i32,
    pub latency: i32,
    pub warps: i32,
    pub occupancy: i32,
    pub cluster: i32,
}

impl GemmTileConfig {
    /// A static launch policy: two persistent blocks per SM, the other knobs left to the compiler.
    pub(super) const fn policy(bm: i32, map_m: i32, map_n: i32) -> Self {
        Self {
            bm,
            map_m,
            map_n,
            blocks_per_sm: 2,
            latency: 0,
            warps: 0,
            occupancy: 0,
            cluster: 0,
        }
    }

    pub(super) fn to_config(self) -> cutile::tune::Config {
        config([
            ("bm", i64::from(self.bm)),
            ("map_m", i64::from(self.map_m)),
            ("map_n", i64::from(self.map_n)),
            ("blocks_per_sm", i64::from(self.blocks_per_sm)),
            ("latency", i64::from(self.latency)),
            ("warps", i64::from(self.warps)),
            ("occupancy", i64::from(self.occupancy)),
            ("cluster", i64::from(self.cluster)),
        ])
    }

    pub(super) fn from_config(config: &cutile::tune::Config) -> Option<Self> {
        let int = |key: &str| config.int(key).and_then(|value| i32::try_from(value).ok());
        Some(Self {
            bm: int("bm")?,
            map_m: int("map_m")?,
            map_n: int("map_n")?,
            blocks_per_sm: int("blocks_per_sm")?,
            latency: int("latency")?,
            warps: int("warps")?,
            occupancy: int("occupancy")?,
            cluster: int("cluster")?,
        })
    }

    pub(super) fn compile_options(self) -> CompileOptions {
        let mut options = CompileOptions::new();
        if self.warps > 0 {
            options = options.num_worker_warps_per_cta(self.warps);
        }
        if self.occupancy > 0 {
            options = options.occupancy(self.occupancy);
        }
        if self.cluster > 0 {
            options = options.num_cta_in_cga(self.cluster);
        }
        options
    }
}

/// The row tiles and swizzle maps to try, with the knobs every FP8 GEMM sweeps around `policy`.
pub(super) fn tile_space(
    bms: impl IntoIterator<Item = [i64; 1]>,
    maps: impl IntoIterator<Item = [i64; 2]>,
    policy: GemmTileConfig,
) -> Space {
    Space::new()
        .joint(["bm"], bms)
        .joint(["map_m", "map_n"], maps)
        .axis("blocks_per_sm", [2, 1, 4])
        .axis("latency", [0, 2, 4])
        .axis("warps", [0, 4, 8])
        .axis("occupancy", [0, 4])
        .axis("cluster", [0, 2])
        .policy(policy.to_config())
}
