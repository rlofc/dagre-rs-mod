// Vendored crate, checked as part of this workspace — `FeasibleTreeState`
// (layout/rank/feasible_tree.rs) is an alternate ranking implementation not
// currently wired into `layout::rank`'s dispatch. Silenced rather than left
// to warn on every `cargo check` here, same as this repo's own genuinely
// scaffolded-for-later dead code.
#![allow(dead_code)]

pub mod graph;
pub mod layout;
pub mod util;

pub use layout::acyclic;
pub use layout::intersect;
pub use layout::{layout, layout_phase1, layout_phase2, layout_phase3};
pub use layout::rank::network_simplex::NetworkSimplexState;
pub use layout::order::OrderState;
pub use layout::position::bk::PositionState;
pub use layout::types::{
    Acyclicer, Align, EdgeLabel, GraphLabel, LayoutOptions, NodeLabel, Point, RankAlign, RankDir,
    Ranker,
};
pub use layout::util::{as_non_compound_graph, build_layer_matrix, normalize_ranks, remove_empty_ranks};
