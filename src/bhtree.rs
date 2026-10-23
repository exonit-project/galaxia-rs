// SPDX-License-Identifier: AGPL-3.0-or-later
// Original C++ code Copyright (c) 2012 Sanjib Sharma and Joss Bland-Hawthorn
// Ported from BHTree.h and BHTree.cpp
//! Barnes-Hut octree reader and leaf manager.
//!
//! Matches Galaxia 0.7.2's `BHTree.h`, `BHTree.cpp`, and `StellarPopulation.cpp`.

use crate::ebf::{EbfError, EbfFile};
use std::path::Path;

/// Octree node representing a spatial-temporal cell in the galaxy.
/// Matches `BHNode` in `BHTree.h:20-166`.
#[derive(Debug, Clone, PartialEq)]
pub struct BHNode {
    /// Spatial and temporal coordinates:
    /// `pos[0..2] = (x, y, z)` in kpc, `pos[3] = age` in yr.
    pub pos: [f64; 4],
    /// Half-side box width in kpc.
    pub l: f64,
    /// Integrated stellar mass of the node in solar masses ($M_\odot$).
    pub mass: f64,
    /// Relative mass error estimate.
    pub delta_mass: f64,
    /// Maximum density within the bounding box in $M_\odot / \mathrm{kpc}^3$.
    pub rho_max: f64,
    /// Minimum density within the bounding box in $M_\odot / \mathrm{kpc}^3$.
    pub rho_min: f64,
    /// Distance modulus from observer: $\mu = 5 \log_{10}(d_{\mathrm{kpc}} \cdot 100.0)$.
    pub dmod: f64,
    /// Half-width in age in yr.
    pub lage: f64,
    /// Pointer/index offset to first child octant.
    /// `-1` (or `0`) indicates a leaf node; $\ge 0$ indicates index of first child.
    pub left: i64,
}

impl Default for BHNode {
    fn default() -> Self {
        Self {
            pos: [0.0; 4],
            l: 0.0,
            mass: 0.0,
            delta_mass: 0.0,
            rho_max: 0.0,
            rho_min: 0.0,
            dmod: 0.0,
            lage: 0.0,
            left: -1,
        }
    }
}

impl BHNode {
    /// Unpack a node from the 12-element `f64` representation stored in EBF.
    /// Matches `BHNode::unpack` in `BHTree.h:84-101`.
    pub fn unpack(data: &[f64]) -> Self {
        assert!(data.len() >= 12, "BHNode record requires 12 doubles");
        let left_raw = data[11].to_bits() as i64;
        Self {
            pos: [data[0], data[1], data[2], data[3]],
            l: data[4],
            mass: data[5],
            delta_mass: data[6],
            rho_max: data[7],
            rho_min: data[8],
            dmod: data[9],
            lage: data[10],
            left: left_raw,
        }
    }

    /// Pack a node into 12 doubles for EBF serialization.
    pub fn pack(&self) -> [f64; 12] {
        [
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.pos[3],
            self.l,
            self.mass,
            self.delta_mass,
            self.rho_max,
            self.rho_min,
            self.dmod,
            self.lage,
            f64::from_bits(self.left as u64),
        ]
    }

    /// Scale the node's mass and density by a factor (e.g. population density correction `rho_fac`).
    /// Matches `BHNode::scale_mass` in `BHTree.h:46-52`.
    #[inline]
    pub fn scale_mass(&mut self, fac: f64) {
        self.mass *= fac;
        self.delta_mass *= fac;
        self.rho_max *= fac;
        self.rho_min *= fac;
    }

    /// Calculate distance modulus from observer position `pos_c`.
    /// Matches `BHNode::calculateDmod(Pos1)` in `BHTree.h:147-151`.
    #[inline]
    pub fn calculate_dmod(&mut self, pos_c: &[f64; 3]) {
        let dx = self.pos[0] - pos_c[0];
        let dy = self.pos[1] - pos_c[1];
        let dz = self.pos[2] - pos_c[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        self.dmod = 5.0 * (dist * 100.0).log10();
    }

    /// Check if this node is a leaf (no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == -1 || self.left == 0
    }

    /// The quirk from Galaxia 0.7.2 (`StellarPopulation.cpp:87`):
    /// ```cpp
    /// if ((ic.n_tot == 1) && (leaf->mass < 1e-2) && (leaf->rho_max / leaf->rho_min > 250.0))
    ///     ic.n_tot = 0;
    /// ```
    /// Suppresses single spurious stars generated in low-mass edge cells with extreme density gradients.
    #[inline]
    pub fn should_suppress_single_star(&self, n_tot: usize) -> bool {
        if n_tot != 1 || self.mass >= 1e-2 {
            return false;
        }
        if self.rho_min <= 0.0 {
            return true;
        }
        (self.rho_max / self.rho_min) > 250.0
    }

    /// Apply the single-star suppression quirk to `n_tot`.
    #[inline]
    pub fn filter_n_tot(&self, n_tot: usize) -> usize {
        if self.should_suppress_single_star(n_tot) {
            0
        } else {
            n_tot
        }
    }
}

/// Ordering comparison function for BHNode leaves by distance modulus.
/// Matches `nodep_less_than` in `BHTree.h:317-322`.
#[inline]
pub fn nodep_less_than(a: &BHNode, b: &BHNode) -> std::cmp::Ordering {
    a.dmod.total_cmp(&b.dmod)
}

/// Barnes-Hut octree holding all nodes and active leaves.
#[derive(Debug, Clone)]
pub struct BHTree {
    /// All tree nodes loaded from `/Nodes` dataset.
    pub nodes: Vec<BHNode>,
    /// Active leaf nodes, with `dmod` calculated and sorted ascending by `dmod`.
    pub leafs: Vec<BHNode>,
}

impl BHTree {
    /// Open and read a BHTree EBF file (`bhtree_{popID}_E{0|1}.ebf`).
    ///
    /// Reads `/Nodes` (N x 12 f64 array) and `/Leafs` (N_leafs i64 index array),
    /// initializes `dmod` for each leaf from `pos_c`, and sorts leaves by `dmod`.
    pub fn load_from_ebf(path: impl AsRef<Path>, pos_c: &[f64; 3]) -> Result<Self, EbfError> {
        let mut ebf = EbfFile::open(path.as_ref())?;

        let nodes_raw = ebf.read_f64("/Nodes")?;
        let n_nodes = nodes_raw.len() / 12;
        let mut nodes = Vec::with_capacity(n_nodes);
        for chunk in nodes_raw.chunks_exact(12) {
            nodes.push(BHNode::unpack(chunk));
        }

        let leaf_indices = ebf.read_i64("/Leafs")?;
        let mut leafs = Vec::with_capacity(leaf_indices.len());
        for &idx in &leaf_indices {
            if idx < 0 || (idx as usize) >= nodes.len() {
                return Err(EbfError::Custom(format!(
                    "Invalid BHTree leaf index {} (nodes length: {})",
                    idx,
                    nodes.len()
                )));
            }
            let uidx = idx as usize;
            let mut leaf = nodes[uidx].clone();
            leaf.calculate_dmod(pos_c);
            leafs.push(leaf);
        }

        // Sort leaves by dmod ascending (matches std::sort(&leafs[0], &leafs[leafs.size()], nodep_less_than()))
        leafs.sort_by(nodep_less_than);

        Ok(Self { nodes, leafs })
    }

    /// Construct BHTree file path and load for a given population ID.
    ///
    /// Follows Galaxia filename convention:
    /// - `warp_flare_on > 0`: `BHTree-2.2/bhtree_with_wf/bhtree_{pop_id}_E{option_e}.ebf`
    /// - `warp_flare_on == 0`: `BHTree-2.2/bhtree_no_wf/bhtree_{pop_id}_E{option_e}.ebf`
    pub fn load_for_pop(
        data_dir: impl AsRef<Path>,
        pop_id: usize,
        option_e: usize,
        warp_flare_on: usize,
        pos_c: &[f64; 3],
    ) -> Result<Self, EbfError> {
        let subdir = if warp_flare_on > 0 {
            "BHTree-2.2/bhtree_with_wf"
        } else {
            "BHTree-2.2/bhtree_no_wf"
        };
        let fname = format!("bhtree_{}_E{}.ebf", pop_id, option_e);
        let path = data_dir.as_ref().join(subdir).join(fname);
        Self::load_from_ebf(path, pos_c)
    }

    /// Get total active leaf count.
    #[inline]
    pub fn num_leafs(&self) -> usize {
        self.leafs.len()
    }

    /// Get total node count.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_pack_unpack() {
        let mut node = BHNode::default();
        node.pos = [1.2, -3.4, 0.56, 1.5e9];
        node.l = 0.05;
        node.mass = 1234.5;
        node.delta_mass = 0.01;
        node.rho_max = 100.0;
        node.rho_min = 1.0;
        node.dmod = 14.25;
        node.lage = 0.02e9;
        node.left = -1;

        let packed = node.pack();
        let unpacked = BHNode::unpack(&packed);
        assert_eq!(node, unpacked);
    }

    #[test]
    fn test_node_quirk_suppression() {
        let mut node = BHNode::default();
        node.mass = 0.005; // < 1e-2
        node.rho_max = 300.0;
        node.rho_min = 1.0; // ratio 300 > 250

        // If n_tot == 1, suppressed to 0
        assert_eq!(node.filter_n_tot(1), 0);
        // If n_tot == 2, not suppressed
        assert_eq!(node.filter_n_tot(2), 2);

        // If mass >= 1e-2, not suppressed
        node.mass = 0.02;
        assert_eq!(node.filter_n_tot(1), 1);

        // If contrast ratio <= 250, not suppressed
        node.mass = 0.005;
        node.rho_max = 200.0;
        node.rho_min = 1.0;
        assert_eq!(node.filter_n_tot(1), 1);
    }

    #[test]
    fn test_load_real_bhtree_ebf() {
        let data_dir = crate::require_data_dir!();
        let path = data_dir.join("BHTree-2.2/bhtree_with_wf/bhtree_8_E0.ebf"); // Spheroid (smallest)
        assert!(path.exists(), "bhtree file not found: {}", path.display());

        let pos_c = [-8.0, 0.0, 0.015];
        let tree = BHTree::load_from_ebf(&path, &pos_c).expect("Failed to load bhtree_8_E0.ebf");

        assert_eq!(tree.num_nodes(), 961289);
        assert_eq!(tree.num_leafs(), 841128);

        // Verify leaves are sorted ascending by dmod
        for i in 1..tree.leafs.len() {
            assert!(
                tree.leafs[i].dmod >= tree.leafs[i - 1].dmod,
                "Leaves must be sorted ascending by dmod"
            );
        }
    }
}
