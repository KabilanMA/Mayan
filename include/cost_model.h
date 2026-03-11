#pragma once
#include "ast.h"
#include "hll.h"
#include "format_selector.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <memory>
#include <cmath>
#include <numeric>
#include <functional>


// FREE: appears in the output; forms the outer loop nest
// CONTRACTED: summed over; forms the inner loop nest; not in the output
// classifies each loop index for this kernel
enum class IndexRole { FREE, CONTRACTED };

struct IndexClassification {
    std::vector<Index> free_indices;        // must appear in output
    std::vector<Index> contracted_indices;  // eliminated by this kernel
};

// NnzEstimationMode - records which path estimate_output_nnz actually took.
// Stored in KernelCostBreakdown so callers can inspect / log the decision.
enum class NnzEstimationMode {
    // No contracted indices exist → pure Cartesian product of all operand NNZs.
    // Formula: Π NNZ(operand_i)
    OUTER_PRODUCT,

    // All operands form a single connected component in the contraction graph
    // (every operand shares at least one contracted index with at least one other).
    // HLL intersection is applied across all operands.
    HLL_SINGLE_COMPONENT,

    // The contraction graph has ≥2 connected components. HLL intersection is
    // applied within each component; results are multiplied across components
    // (Cartesian product between disconnected sub-expressions).
    HLL_MULTI_COMPONENT,

    // Not all operands carry HLL sketches -> fell back to density × output_volume.
    DENSITY_PRODUCT_FALLBACK,
};

// KernelCostBreakdown — detailed cost components returned to the DP
struct KernelCostBreakdown {
    double compute_cost      = 0.0;  // FLOPs proxy: Σ NNZ of operands × divergence factor
    double access_penalty    = 0.0;  // CSF discordant-access penalty (depth-scaled)
    double memory_write_cost = 0.0;  // HBM write cost for output materialization
    double total_cost        = 0.0;
    NnzEstimationMode nnz_mode = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
};

// FusedCostResult — everything the DP needs to know about a proposed fusion
struct FusedCostResult {
    double total_cost;
    double estimated_out_nnz;
    std::vector<Index> loop_order;     // full loop nest: free first, contracted inner
    StorageFormat out_format;
    KernelCostBreakdown breakdown;      // for debugging / tuning

    // One entry per operand, same order as the operands vector passed in.
    // For TensorNode leaves: gives the CSR/CSC/CSF recommendation plus
    //   the COO re-sort cost absorbed into total_cost.
    // For FusedContractionNode intermediates: reflects the existing output format.
    std::vector<InputFormatRecommendation> input_formats;
};

/**
 * @class CostModel
 * @brief Calculate the cost for different optimization and fusion for the kernel
 * Three responsibilities:
 *      1. Index classification  — which indices are free vs. contracted.
 *      2. NNZ estimation        — uses HLL sketches when available.
 *      3. Kernel cost scoring   — compute + access + write costs.
 */
class CostModel {
public:

    /**
     * @brief Evaluate the total cost of fusing a subset of operands into one kernel.
     * 
     * @param operands the ExprNodes being fused (TensorNode or FusedContractionNode)
     * @param output_indices indices that must survive this contraction
     * @param dim_sizes dimension sizes keyed by Index (for sparsity calculation)
     * 
     * @return The cost information for the proposed fusion
     */
    static FusedCostResult evaluate_fused(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        if (operands.empty()) {
            return {0.0, 0.0, {}, {}, {}, {}};
        }

        // Classify indices as FREE (outer loops) vs CONTRACTED (inner)
        const IndexClassification idx_class = classify_indices(operands, output_indices);

        // Estimate output NNZ using HLL sketches when available, 
        // otherwise fall back to the density-product heuristic.
        NnzEstimationMode nnz_mode = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
        const double estimated_out_nnz = estimate_output_nnz(operands, output_indices, idx_class, dim_sizes, nnz_mode);

        // ------------------------------------------------------------------
        // Step 3: Determine loop order.
        //         Phase A — free indices first (outermost), ordered by
        //                   NNZ-weighted concordance to maximise cache reuse
        //                   on the output tensor.
        //         Phase B — contracted indices innermost, ordered by
        //                   concordance to keep inner loops sequential.
        // ------------------------------------------------------------------
        const std::vector<Index> loop_order = determine_loop_order(operands, idx_class);

        // ------------------------------------------------------------------
        // Step 4: Choose the output CSF storage format.
        //         Mode order mirrors the free-index loop order to guarantee
        //         concordant writes (no post-transpose needed for the next
        //         DP stage).
        // ------------------------------------------------------------------
        const StorageFormat out_format = choose_output_format(loop_order, output_indices, estimated_out_nnz, dim_sizes);

        // ------------------------------------------------------------------
        // Step 5: Score the kernel
        // ------------------------------------------------------------------
        KernelCostBreakdown breakdown = score_kernel(operands, loop_order, estimated_out_nnz);
        breakdown.nnz_mode = nnz_mode;

        // ------------------------------------------------------------------
        // Step 6: Recommend input formats (CSR / CSC / CSF) and account for
        //         COO → CSF conversion cost.
        //
        //         Conversion cost is a one-time sort: NNZ × log(NNZ).
        //         It is included in total_cost so the DP can correctly prefer
        //         a plan where inputs need no re-sorting over one that is
        //         cheaper to execute but requires expensive COO conversion.
        // ------------------------------------------------------------------
        std::vector<InputFormatRecommendation> input_formats = FormatSelector::recommend_all(operands, loop_order, dim_sizes);

        double coo_conversion_cost = 0.0;
        for (size_t i = 0; i < operands.size(); ++i) {
            if (auto t = std::dynamic_pointer_cast<TensorNode>(operands[i])) {
                // Only pay conversion cost if the recommended format differs
                // from the tensor's current physical layout
                if (FormatSelector::needs_reformat(*t, input_formats[i].format.mode_order)) {
                    coo_conversion_cost += FormatSelector::conversion_cost(*t);
                }
            }
        }

        const double total_cost = breakdown.total_cost + coo_conversion_cost;

        return {total_cost, estimated_out_nnz, loop_order, out_format, breakdown, std::move(input_formats)};
    }

    // -------------------------------------------------------------------------
    // Infer which indices of a subset must appear in its output.
    //
    // Rule:  Output(S) = I(S) ∩ ( I(S^c) ∪ G_out )
    //
    //   I(S)   = all indices that appear in any operand in subset S
    //   I(S^c) = all indices that appear in any operand OUTSIDE S
    //   G_out  = the final global output indices of the whole expression
    //
    // An index must survive if it is needed by a later stage (I(S^c)) or
    // must appear in the final answer (G_out).
    // -------------------------------------------------------------------------
    static std::vector<Index> infer_output_indices(
        const std::vector<std::shared_ptr<ExprNode>>& subset_operands,
        const std::vector<std::shared_ptr<ExprNode>>& all_operands,
        const std::vector<Index>& global_out_indices)
    {
        std::unordered_set<Index> subset_idx;
        std::unordered_set<Index> remaining_idx;
        const std::unordered_set<Index> global_out(global_out_indices.begin(), global_out_indices.end());

        // I(S)
        for (const auto& op : subset_operands) {
            for (Index idx : op->get_indices()) {
                subset_idx.insert(idx);
            }
        }

        // I(S^c) — identify remaining operands by stable pointer identity
        // (callers must ensure pointer identity is preserved across DP states)
        const std::unordered_set<const ExprNode*> subset_ptrs = [&] {
            std::unordered_set<const ExprNode*> s;
            for (const auto& op : subset_operands) 
                s.insert(op.get());
            return s;
        }();

        for (const auto& op : all_operands) {
            if (subset_ptrs.count(op.get()) == 0) {
                for (Index idx : op->get_indices()) {
                    remaining_idx.insert(idx);
                }
            }
        }

        // Intersection rule
        std::vector<Index> output;
        output.reserve(subset_idx.size());
        for (Index idx : subset_idx) {
            if (remaining_idx.count(idx) || global_out.count(idx)) {
                output.push_back(idx);
            }
        }

        // Canonical sort: stable DP memoisation keys
        std::sort(output.begin(), output.end());
        return output;
    }

    /**
     * @brief Classify all indices appearing in `operands` into FREE vs CONTRACTED.
     * 
     * @param operands Tensors to consider for the contraction.
     * @param output_indices 
     * 
     * @return vectors of free and contracted indices
     */
    static IndexClassification classify_indices(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices)
    {
        const std::unordered_set<Index> out_set(output_indices.begin(), output_indices.end());

        std::unordered_set<Index> all_set;
        for (const auto& op : operands) {
            for (Index idx : op->get_indices()) 
                all_set.insert(idx);
        }

        IndexClassification result;
        for (Index idx : all_set) {
            if (out_set.count(idx)) {
                result.free_indices.push_back(idx);
            } else {
                result.contracted_indices.push_back(idx);
            }
        }

        // Stable canonical order for reproducibility
        std::sort(result.free_indices.begin(), result.free_indices.end());
        std::sort(result.contracted_indices.begin(), result.contracted_indices.end());
        return result;
    }

private:
    // =========================================================================
    // NNZ Estimation
    // =========================================================================

    // Estimate the number of non-zeros in the output of this kernel.
    //
    // Strategy:
    //  (a) If ALL operands carry non-empty HLL sketches, delegate to
    //      estimate_nnz_via_hll which applies connected-component analysis
    //      to pick the right formula per case (see NnzEstimationMode).
    //  (b) Otherwise, fall back to a density-product heuristic:
    //        density(output) ≈ Π density(operand_i)
    //        output_volume   = Π dim_size(free_index_j)
    //        estimated_nnz   = density(output) × output_volume
    //
    /**
     * @brief Estimate the number of non-zeros in the output of this kernel.
     * 
     * Strategy:
     * 
     * (a) If ALL operands carry non-empty HLL sketches, delegate to
     *      estimate_nnz_via_hll which applies connected-component anal
     *      to pick the right formula per case (see NnzEstimationMode).
     * 
     * (b) Otherwise, fall back to a density-product heuristic:
     * 
     *      - density(output) ≈ Π density(operand_i)
     * 
     *      - output_volume   = Π dim_size(free_index_j)
     * 
     *      - estimated_nnz   = density(output) × output_volume
     */
    static double estimate_output_nnz(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices,
        const IndexClassification& idx_class,
        const std::unordered_map<Index, int>& dim_sizes,
        NnzEstimationMode& mode_out)
    {
        // Attempt HLL path
        if (all_have_hll_sketches(operands)) {
            return estimate_nnz_via_hll(operands, idx_class, mode_out);
        }
        mode_out = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
        // Fallback density-product path
        return estimate_nnz_density_product(operands, output_indices, dim_sizes);
    }

    static bool all_have_hll_sketches(const std::vector<std::shared_ptr<ExprNode>>& operands)
    {
        for (const auto& op : operands) {
            if (op->metadata.hll_sketch.empty()) 
                return false;
        }
        return true;
    }

        // =========================================================================
    // estimate_nnz_via_hll — the corrected HLL path
    //
    // WHY THE OLD CODE WAS WRONG
    // ──────────────────────────
    // Each tensor's HLL sketch hashes its *full coordinate tuple*.
    // B(i,k) hashes (i,k) pairs  →  lives in coordinate space ℤᵢ × ℤₖ
    // C(k,j) hashes (k,j) pairs  →  lives in coordinate space ℤₖ × ℤⱼ
    //
    // Calling estimate_intersection(B_sketch, C_sketch) computes |B ∩ C| in
    // the *hash* space.  Because B and C live in DIFFERENT coordinate spaces,
    // this intersection has no mathematical meaning — it returns a small number
    // by random hash collision, not by actual sparsity structure.
    //
    // THE THREE CASES AND THEIR CORRECT FORMULAS
    // ──────────────────────────────────────────
    //
    // Case 1 — No contracted indices (pure outer product):
    //   A(i,j) ⊗ B(k,l) → C(i,j,k,l)
    //   Every NNZ of A paired with every NNZ of B produces a NNZ in C.
    //   Output NNZ = Π NNZ(operandᵢ)
    //   Intersection is completely wrong here.
    //
    // Case 2 — Single connected component:
    //   B(i,k) * C(k,j) → A(i,j)   [k is contracted, connects B and C]
    //   Both operands share contracted index k.  Their sketches are built
    //   over the same k-coordinate domain, making intersection a reasonable
    //   proxy.  We apply HLL intersection to the whole group.
    //
    // Case 3 — Multiple connected components:
    //   A(i,j) * B(k,l) * C(j,k) → D(i,l)
    //   Contraction graph: A──C──B (j connects A-C, k connects C-B).
    //   A, B, C are all connected → single component here.
    //   But consider: A(i,j) * B(j,k) * C(l,m) → D(i,k,l,m)
    //   Graph: A──B   C  (C is isolated; j is contracted, l and m are free)
    //   Component 1: {A, B} → intersection estimate
    //   Component 2: {C}    → C's own NNZ
    //   Output NNZ = NNZ(A*B) × NNZ(C)   [Cartesian product across components]
    //
    // BUILDING THE CONTRACTION GRAPH
    // ──────────────────────────────
    // Nodes  = operands (indices 0..N-1)
    // Edge(i,j) iff operands i and j share ≥1 contracted index.
    // Connected components are found via union-find.
    // =========================================================================
    /**
     * @brief     // HLL-based estimation:
    // The expected output NNZ = intersection_cardinality(contracted dims) ×
    //                           product_of_free_dim_densities
    //
    // In practice we estimate:
    //   nnz ≈ min( Π |op_i| ,  Σ|op_i| ) bounded by output volume
    //
    // The n-way intersection of the HLL sketches gives a sharper lower-bound
    // than the raw min() used in earlier versions.
     */
    static double estimate_nnz_via_hll(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const IndexClassification& idx_class,
        NnzEstimationMode& mode_out)
    {
        const int N = static_cast<int>(operands.size());

        // ── Case 1: no contracted indices -> pure outer product ─────────────────
        if (idx_class.contracted_indices.empty()) {
            mode_out = NnzEstimationMode::OUTER_PRODUCT;
            double product = 1.0;
            for (const auto& op : operands) product *= op->estimate_nnz();
            return std::max(1.0, product);
        }

        // ── Build contraction graph ────────────────────────────────────────────
        // For each operand, collect which contracted indices it owns.
        const std::unordered_set<Index> contracted_set(
            idx_class.contracted_indices.begin(),
            idx_class.contracted_indices.end());

        std::vector<std::unordered_set<Index>> op_contracted(N);
        for (int i = 0; i < N; ++i) {
            for (Index idx : operands[i]->get_indices()) {
                if (contracted_set.count(idx)) op_contracted[i].insert(idx);
            }
        }

        // Union-find
        std::vector<int> parent(N);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<int(int)> find = [&](int x) -> int {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };
        auto unite = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a != b) parent[a] = b;
        };

        // Connect operands that share a contracted index.
        // NOTE: operands that are only connected through FREE indices are NOT
        // joined here — free indices don't create a join dependency; they just
        // mean both tensors are iterated over in the same outer loop.
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                for (Index idx : op_contracted[i]) {
                    if (op_contracted[j].count(idx)) { unite(i, j); break; }
                }
            }
        }

        // Group operands by component root
        std::unordered_map<int, std::vector<int>> components;
        for (int i = 0; i < N; ++i) components[find(i)].push_back(i);

        // ── Reconstruct sketches ───────────────────────────────────────────────
        std::vector<HyperLogLog> sketches;
        sketches.reserve(N);
        for (const auto& op : operands) sketches.emplace_back(op->metadata.hll_sketch);

        // ── Compute NNZ per component, multiply across components ──────────────
        //
        // Within a component: HLL intersection is valid because all operands
        // share at least one contracted index — their coordinate spaces overlap
        // in the k-dimension, making the intersection a useful order-of-magnitude
        // proxy for the number of matching k-values.
        //
        // Across components: pure Cartesian product (different index spaces).
        double total_nnz = 1.0;

        for (const auto& [root, members] : components) {
            double component_nnz;
            if (static_cast<int>(members.size()) == 1) {
                // Isolated operand: no contracted index shared with anyone else.
                // Its full NNZ is its contribution to the outer product.
                component_nnz = sketches[members[0]].estimate();
            } else {
                // Connected component: intersection estimate
                std::vector<const HyperLogLog*> ptrs;
                ptrs.reserve(members.size());
                for (int idx : members) ptrs.push_back(&sketches[idx]);
                component_nnz = HyperLogLog::estimate_intersection(ptrs);
            }
            total_nnz *= std::max(1.0, component_nnz);
        }

        mode_out = (components.size() == 1)
            ? NnzEstimationMode::HLL_SINGLE_COMPONENT
            : NnzEstimationMode::HLL_MULTI_COMPONENT;

        return std::max(1.0, total_nnz);
    }

    // Density-product fallback:
    //   density_output ≈ Π density_i
    //   out_volume      = Π dim_size[idx] for free indices
    //   nnz             = density_output × out_volume
    static double estimate_nnz_density_product(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      output_indices,
        const std::unordered_map<Index, int>&          dim_sizes)
    {
        // Accumulate product of per-operand densities
        double density_product = 1.0;
        for (const auto& op : operands) {
            density_product *= op->metadata.global_density;
        }
        density_product = std::max(density_product, 1e-9); // avoid underflow

        // Output volume = Π dim_size for each free (output) index
        double out_volume = 1.0;
        for (Index idx : output_indices) {
            auto it = dim_sizes.find(idx);
            if (it != dim_sizes.end()) {
                out_volume *= static_cast<double>(it->second);
            } else {
                // No size info: use the operand's raw NNZ as a conservative bound
                double max_nnz = 0.0;
                for (const auto& op : operands) {
                    max_nnz = std::max(max_nnz, op->estimate_nnz());
                }
                out_volume *= std::sqrt(max_nnz); // rough geometric mean
            }
        }

        return std::max(1.0, density_product * out_volume);
    }

    // =========================================================================
    // Loop Order — Two-Phase NNZ-Weighted Concordance Sort
    // =========================================================================

    // Phase A: Sort FREE indices by descending NNZ-weighted concordance vote.
    //          Placing the index that the largest tensor wants first maximises
    //          cache-line reuse when streaming through the output.
    // Phase B: Sort CONTRACTED indices the same way.
    //          Innermost contracted loops benefit from register reuse
    //          (the partial sum accumulator stays in a register).
    //
    // The "concordance vote" for index `k` is: Σ_i { nnz_i  if  k is the first
    // unassigned index in tensor i's physical mode order, else 0 }.
    // This greedily maximises the number of tensors accessed concordantly.
    static std::vector<Index> determine_loop_order(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const IndexClassification&                     idx_class)
    {
        std::vector<Index> order;
        order.reserve(idx_class.free_indices.size() +
                      idx_class.contracted_indices.size());

        // Phase A: free indices (outer loops)
        auto free_ordered = greedy_concordance_sort(
            operands,
            std::unordered_set<Index>(idx_class.free_indices.begin(),
                                      idx_class.free_indices.end()));
        order.insert(order.end(), free_ordered.begin(), free_ordered.end());

        // Phase B: contracted indices (inner loops)
        auto contr_ordered = greedy_concordance_sort(
            operands,
            std::unordered_set<Index>(idx_class.contracted_indices.begin(),
                                      idx_class.contracted_indices.end()));
        order.insert(order.end(), contr_ordered.begin(), contr_ordered.end());

        return order;
    }

    // Core greedy concordance sort over a given candidate set.
    static std::vector<Index> greedy_concordance_sort(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        std::unordered_set<Index>                      candidates)
    {
        std::vector<Index> result;
        result.reserve(candidates.size());

        while (!candidates.empty()) {
            std::unordered_map<Index, double> votes;
            for (Index idx : candidates) votes[idx] = 0.0;

            for (const auto& op : operands) {
                const std::vector<Index> mode_order = get_mode_order(op);
                // Find the first unassigned index in this tensor's physical layout
                for (Index idx : mode_order) {
                    if (candidates.count(idx)) {
                        votes[idx] += op->estimate_nnz();
                        break; // only the first unassigned index votes
                    }
                }
            }

            // Select the index with the highest concordance score
            Index best = candidates.begin().operator*();
            double best_score = -1.0;
            for (const auto& [idx, score] : votes) {
                if (score > best_score) { best_score = score; best = idx; }
            }

            result.push_back(best);
            candidates.erase(best);
        }

        return result;
    }

    // =========================================================================
    // Output CSF Format Selection
    // =========================================================================

    // Rules:
    //  - Mode order mirrors the free-index loop order: concordant writes,
    //    no transpose needed before the next DP stage.
    //  - Innermost level: DENSE if density > 30% (enables SIMD vectorisation);
    //    COMPRESSED otherwise (memory efficiency).
    //  - All other levels: always COMPRESSED (standard CSF).
    static StorageFormat choose_output_format(
        const std::vector<Index>&             loop_order,
        const std::vector<Index>&             output_indices,
        double                                estimated_out_nnz,
        const std::unordered_map<Index, int>& dim_sizes)
    {
        // Extract the free-index subsequence of the loop order
        const std::unordered_set<Index> out_set(
            output_indices.begin(), output_indices.end());

        std::vector<Index> mode_order;
        mode_order.reserve(output_indices.size());
        for (Index idx : loop_order) {
            if (out_set.count(idx)) mode_order.push_back(idx);
        }

        // Compute actual output tensor volume from dim_sizes when available
        double out_volume = 1.0;
        for (Index idx : mode_order) {
            auto it = dim_sizes.find(idx);
            out_volume *= (it != dim_sizes.end())
                ? static_cast<double>(it->second)
                : 100.0; // fallback if sizes unknown
        }
        const double sparsity = (out_volume > 0.0)
            ? (estimated_out_nnz / out_volume)
            : 0.0;

        std::vector<LevelType> level_types;
        level_types.reserve(mode_order.size());
        for (size_t i = 0; i < mode_order.size(); ++i) {
            const bool is_innermost = (i == mode_order.size() - 1);
            // DENSE innermost: only if the output is dense enough to fill a
            // SIMD register without wasting bandwidth on zeroes
            level_types.push_back(
                (is_innermost && sparsity > 0.30)
                ? LevelType::DENSE
                : LevelType::COMPRESSED);
        }

        return StorageFormat{level_types, mode_order};
    }

    // =========================================================================
    // Kernel Cost Scoring
    // =========================================================================

    // The cost model has three additive terms:
    //
    //  1. Compute cost   ≈ Σ NNZ_i  scaled by a branch-divergence factor.
    //     Branch divergence arises from coordinating sparse fibers from N
    //     different CSF trees in lock-step; it grows roughly as O(log N).
    //
    //  2. Access penalty: applied per operand whose physical CSF mode order
    //     is discordant with the chosen loop order.  The penalty is EXPONENTIAL
    //     in the inversion depth: depth 1 (2nd before 1st mode) is 5×; depth 2
    //     (3rd before 1st) is 25×; etc. This correctly models the difference
    //     between a single binary-search per fiber (depth 1) vs. a full linear
    //     scan of the entire level (depth 2+).
    //
    //  3. Memory write cost ≈ estimated_out_nnz × HBM_WRITE_FACTOR.
    //     Materialising the intermediate result to HBM is typically the
    //     bottleneck on bandwidth-limited accelerators (TPUs, GPUs).
    //
    static KernelCostBreakdown score_kernel(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      loop_order,
        double                                         estimated_out_nnz)
    {
        // Tuneable cost constants (normalised to HBM bandwidth units)
        constexpr double DISCORDANT_PENALTY_BASE  = 5.0;   // per depth level
        constexpr double HBM_WRITE_FACTOR         = 1.5;   // output write cost
        constexpr double DIVERGENCE_LOG_FACTOR    = 0.15;  // branch divergence

        KernelCostBreakdown result;

        for (const auto& op : operands) {
            const double op_nnz      = op->estimate_nnz();
            result.compute_cost     += op_nnz;
            result.access_penalty   += compute_access_penalty(
                op, loop_order, op_nnz, DISCORDANT_PENALTY_BASE);
        }

        // Branch divergence: log(N) reflects the cost of N-way merge iterators
        // in a co-iteration loop. For N=2 (binary), this is ~0.1. For N=8, ~0.3.
        const double divergence_factor =
            1.0 + DIVERGENCE_LOG_FACTOR * std::log2(
                static_cast<double>(operands.size()) + 1.0);
        result.compute_cost *= divergence_factor;

        result.memory_write_cost = estimated_out_nnz * HBM_WRITE_FACTOR;
        result.total_cost = result.compute_cost
                          + result.memory_write_cost
                          + result.access_penalty;
        return result;
    }

    // Per-operand discordant-access penalty.
    //
    // Access is CONCORDANT if, among the loop indices that belong to this
    // tensor, the tensor's primary mode (mode_order[0]) is encountered FIRST
    // in the loop order.  Any secondary mode encountered before the primary
    // forces an out-of-order traversal.
    //
    // Penalty = NNZ × BASE^(inversion_depth)
    // where inversion_depth = (position of primary mode in loop_order)
    //                       - (position of earliest secondary mode in loop_order
    //                          that appears before the primary)
    static double compute_access_penalty(
        const std::shared_ptr<ExprNode>& op,
        const std::vector<Index>&         loop_order,
        double                            op_nnz,
        double                            base)
    {
        const std::vector<Index> mode_order = get_mode_order(op);
        if (mode_order.size() < 2) return 0.0;

        // Build a position map for the loop order
        std::unordered_map<Index, int> loop_pos;
        for (int i = 0; i < static_cast<int>(loop_order.size()); ++i) {
            loop_pos[loop_order[i]] = i;
        }

        const Index primary = mode_order[0];
        auto prim_it = loop_pos.find(primary);
        if (prim_it == loop_pos.end()) return 0.0; // primary not in this kernel
        const int primary_pos = prim_it->second;

        // Find the earliest secondary mode that appears before the primary
        int earliest_secondary_pos = std::numeric_limits<int>::max();
        for (size_t m = 1; m < mode_order.size(); ++m) {
            auto sec_it = loop_pos.find(mode_order[m]);
            if (sec_it == loop_pos.end()) continue;
            if (sec_it->second < primary_pos) {
                earliest_secondary_pos =
                    std::min(earliest_secondary_pos, sec_it->second);
            }
        }

        if (earliest_secondary_pos == std::numeric_limits<int>::max()) {
            return 0.0; // concordant access
        }

        // Inversion depth: how many loop levels separate the wrongly-placed
        // secondary from where the primary should have been
        const int inversion_depth = primary_pos - earliest_secondary_pos;
        return op_nnz * std::pow(base, static_cast<double>(inversion_depth));
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    static std::vector<Index> get_mode_order(
        const std::shared_ptr<ExprNode>& op)
    {
        if (auto t = std::dynamic_pointer_cast<TensorNode>(op)) {
            return t->format.mode_order;
        }
        if (auto f = std::dynamic_pointer_cast<FusedContractionNode>(op)) {
            return f->output_format.mode_order;
        }
        return {};
    }
};