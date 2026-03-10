#pragma once
#include "ast.h"
#include "hll.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <memory>
#include <cmath>
#include <numeric>

// =============================================================================
// IndexRole — classifies each loop index for this kernel
//
//   FREE        : appears in the output; forms the outer loop nest
//   CONTRACTED  : summed over; forms the inner loop nest; not in the output
// =============================================================================
enum class IndexRole { FREE, CONTRACTED };

struct IndexClassification {
    std::vector<Index> free_indices;        // must appear in output
    std::vector<Index> contracted_indices;  // eliminated by this kernel
};

// =============================================================================
// KernelCostBreakdown — detailed cost components returned to the DP
// =============================================================================
struct KernelCostBreakdown {
    double compute_cost      = 0.0;  // FLOPs proxy: Σ NNZ of operands × divergence factor
    double access_penalty    = 0.0;  // CSF discordant-access penalty (depth-scaled)
    double memory_write_cost = 0.0;  // HBM write cost for output materialization
    double total_cost        = 0.0;
};

// =============================================================================
// FusedCostResult — everything the DP needs to know about a proposed fusion
// =============================================================================
struct FusedCostResult {
    double                total_cost;
    double                estimated_out_nnz;
    std::vector<Index>    loop_order;     // full loop nest: free first, contracted inner
    StorageFormat         out_format;
    KernelCostBreakdown   breakdown;      // for debugging / tuning
};

// =============================================================================
// CostModel
//
// Three responsibilities:
//   1. Index classification  — which indices are free vs. contracted
//   2. NNZ estimation        — uses HLL sketches when available
//   3. Kernel cost scoring   — compute + access + write costs
// =============================================================================
class CostModel {
public:
    // -------------------------------------------------------------------------
    // Evaluate the total cost of fusing a subset of operands into one kernel.
    //
    //   operands        : the ExprNodes being fused (TensorNode or FusedContractionNode)
    //   output_indices  : indices that must survive this contraction
    //   dim_sizes       : dimension sizes keyed by Index (for sparsity calculation)
    // -------------------------------------------------------------------------
    static FusedCostResult evaluate_fused(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        if (operands.empty()) {
            return {0.0, 0.0, {}, {}, {}};
        }

        // ------------------------------------------------------------------
        // Step 1: Classify indices as FREE (outer loops) vs CONTRACTED (inner)
        // ------------------------------------------------------------------
        const IndexClassification idx_class = classify_indices(operands, output_indices);

        // ------------------------------------------------------------------
        // Step 2: Estimate output NNZ using HLL sketches when available,
        //         otherwise fall back to the density-product heuristic.
        // ------------------------------------------------------------------
        const double estimated_out_nnz = estimate_output_nnz(
            operands, output_indices, idx_class, dim_sizes);

        // ------------------------------------------------------------------
        // Step 3: Determine loop order.
        //         Phase A — free indices first (outermost), ordered by
        //                   NNZ-weighted concordance to maximise cache reuse
        //                   on the output tensor.
        //         Phase B — contracted indices innermost, ordered by
        //                   concordance to keep inner loops sequential.
        // ------------------------------------------------------------------
        const std::vector<Index> loop_order = determine_loop_order(
            operands, idx_class);

        // ------------------------------------------------------------------
        // Step 4: Choose the output CSF storage format.
        //         Mode order mirrors the free-index loop order to guarantee
        //         concordant writes (no post-transpose needed for the next
        //         DP stage).
        // ------------------------------------------------------------------
        const StorageFormat out_format = choose_output_format(
            loop_order, output_indices, estimated_out_nnz, dim_sizes);

        // ------------------------------------------------------------------
        // Step 5: Score the kernel
        // ------------------------------------------------------------------
        const KernelCostBreakdown breakdown = score_kernel(
            operands, loop_order, estimated_out_nnz);

        return {breakdown.total_cost, estimated_out_nnz,
                loop_order, out_format, breakdown};
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

    // -------------------------------------------------------------------------
    // Classify all indices appearing in `operands` into FREE vs CONTRACTED.
    // -------------------------------------------------------------------------
    static IndexClassification classify_indices(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      output_indices)
    {
        const std::unordered_set<Index> out_set(
            output_indices.begin(), output_indices.end());

        std::unordered_set<Index> all_set;
        for (const auto& op : operands) {
            for (Index idx : op->get_indices()) all_set.insert(idx);
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
        std::sort(result.free_indices.begin(),       result.free_indices.end());
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
    //  (a) If ALL operands carry non-empty HLL sketches, use the set-theoretic
    //      intersection estimate on the CONTRACTED dimensions. The output NNZ
    //      is then the intersection cardinality projected onto free dimensions.
    //  (b) Otherwise, fall back to a density-product heuristic:
    //        density(output) ≈ Π density(operand_i)
    //        output_volume   = Π dim_size(free_index_j)
    //        estimated_nnz   = density(output) × output_volume
    //
    static double estimate_output_nnz(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      output_indices,
        const IndexClassification&                     idx_class,
        const std::unordered_map<Index, int>&          dim_sizes)
    {
        // Attempt HLL path
        if (all_have_hll_sketches(operands)) {
            return estimate_nnz_via_hll(operands, output_indices, idx_class, dim_sizes);
        }
        // Fallback density-product path
        return estimate_nnz_density_product(operands, output_indices, dim_sizes);
    }

    static bool all_have_hll_sketches(
        const std::vector<std::shared_ptr<ExprNode>>& operands)
    {
        for (const auto& op : operands) {
            if (op->metadata.hll_sketch.empty()) return false;
        }
        return true;
    }

    // HLL-based estimation:
    // The expected output NNZ = intersection_cardinality(contracted dims) ×
    //                           product_of_free_dim_densities
    //
    // In practice we estimate:
    //   nnz ≈ min( Π |op_i| ,  Σ|op_i| ) bounded by output volume
    //
    // The n-way intersection of the HLL sketches gives a sharper lower-bound
    // than the raw min() used in earlier versions.
    static double estimate_nnz_via_hll(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      /*output_indices*/,
        const IndexClassification&                     /*idx_class*/,
        const std::unordered_map<Index, int>&          /*dim_sizes*/)
    {
        // Reconstruct HLL sketches from metadata
        std::vector<HyperLogLog> sketches;
        sketches.reserve(operands.size());
        for (const auto& op : operands) {
            sketches.emplace_back(op->metadata.hll_sketch);
        }

        // Build pointer list for n-way intersection
        std::vector<const HyperLogLog*> sketch_ptrs;
        sketch_ptrs.reserve(sketches.size());
        for (const auto& s : sketches) sketch_ptrs.push_back(&s);

        // N-way intersection gives the expected number of coordinate tuples
        // that survive the join — a tight upper bound on output NNZ.
        const double intersection_est =
            HyperLogLog::estimate_intersection(sketch_ptrs);

        return std::max(1.0, intersection_est);
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
