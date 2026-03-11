#pragma once
#include "ast.h"
#include "cost_model.h"
#include "format_selector.h"
#include <vector>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <string>
#include <functional>
#include <cassert>

// =============================================================================
// DPState — best plan found so far for a given bitmask of input tensors
// =============================================================================
struct DPState {
    double min_cost = std::numeric_limits<double>::max();
    std::shared_ptr<ExprNode> best_ast = nullptr;

    bool is_valid() const { return best_ast != nullptr; }

    // Relax: overwrite if the new plan is strictly cheaper
    bool try_relax(double candidate_cost, std::shared_ptr<ExprNode> candidate_ast) {
        if (candidate_cost < min_cost) {
            min_cost = candidate_cost;
            best_ast  = std::move(candidate_ast);
            return true;
        }
        return false;
    }
};

// =============================================================================
// DPOptimizer
//
// Finds the optimal sparse tensor contraction plan using a bitmask DP over
// subsets of input tensors. For each subset (bitmask), three strategies compete:
//
//   Strategy A — N-ary Fusion
//     Evaluate all tensors in the subset together in a single loop nest.
//     Best when operands are small or share many index dimensions.
//
//   Strategy B — Binary Contraction Sequence
//     Split the subset into two disjoint sub-masks. Recursively use the best
//     plans for each half, then contract the two intermediate results.
//     Best when one half can eliminate many indices, reducing work for the other.
//
//   Strategy C — Pivot Index Decomposition
//     Identify a contracted index k that appears in only a SUBSET of the
//     operands. Partition into G_k (tensors containing k) and G_rest (others).
//     Pre-compute an intermediate tensor T by contracting G_k (eliminating k).
//     Then contract T with G_rest.
//
//     Key insight: G_rest never had to iterate over k at all. By materialising
//     T first, we reduce the loop nest arity for G_rest and avoid redundant
//     traversal of the k dimension.  This is especially powerful for "star"
//     schemas where a hub index k connects a few large tensors.
//
// The DP table has 2^N entries (one per subset). The final answer is at
// dp[all_bits_set].  N must be ≤ 20 for the bitmask approach to be practical;
// for larger N, a greedy or randomised strategy should be substituted.
// =============================================================================
class DPOptimizer {
public:
    static constexpr int MAX_INPUTS = 20; // 2^20 = ~1M DP states

    // -------------------------------------------------------------------------
    // Main entry point.
    //
    //   inputs            : leaf TensorNodes (or already-fused sub-expressions)
    //   global_out_indices: the indices that the final result must have
    //   dim_sizes         : dimension sizes keyed by Index char; passed through
    //                       to the cost model for accurate sparsity estimation
    // -------------------------------------------------------------------------
    static std::shared_ptr<ExprNode> optimize(
        const std::vector<std::shared_ptr<ExprNode>>& inputs,
        const std::vector<Index>& global_out_indices,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        const int N = static_cast<int>(inputs.size());
        assert(N > 0 && N <= MAX_INPUTS);

        const int num_subsets = 1 << N;
        std::vector<DPState> dp(num_subsets);

        // ----- Base cases: single tensors cost nothing to "plan" --------------
        for (int i = 0; i < N; ++i) {
            const int mask = 1 << i;
            dp[mask].min_cost = 0.0;
            dp[mask].best_ast = inputs[i];
        }

        // ----- Main DP loop: process subsets by increasing size ---------------
        for (int size = 2; size <= N; ++size) {
            for (int mask = 1; mask < num_subsets; ++mask) {
                if (__builtin_popcount(mask) != size) 
                    continue;

                // Gather operands present in this subset
                std::vector<std::shared_ptr<ExprNode>> subset_ops;
                subset_ops.reserve(size);
                for (int i = 0; i < N; ++i) {
                    if (mask & (1 << i)) subset_ops.push_back(inputs[i]);
                }

                // Determine which indices this subset must output
                const std::vector<Index> subset_out = CostModel::infer_output_indices(subset_ops, inputs, global_out_indices);

                // ============================================================
                // Strategy A: N-ary Fusion
                // ============================================================
                evaluate_nary_fusion(dp[mask], subset_ops, subset_out, dim_sizes);

                // ============================================================
                // Strategy B: Binary Contraction Sequence
                // ============================================================
                evaluate_binary_splits(dp, mask, subset_out, dim_sizes, inputs,
                                       global_out_indices);

                // ============================================================
                // Strategy C: Pivot Index Decomposition
                // ============================================================
                evaluate_pivot_decomposition(dp, mask, subset_ops, subset_out,
                                             dim_sizes, inputs, global_out_indices);
            }
        }

        assert(dp[num_subsets - 1].is_valid());

        // ── Post-pass: rewrite every leaf TensorNode in the winning AST
        //    to carry the format recommended by its consuming kernel's loop order.
        //    This makes to_string() display the correct physical index order
        //    and enables code-gen to emit the right COO → CSF sort order.
        return apply_recommended_formats(dp[num_subsets - 1].best_ast);
    }

private:
    // =========================================================================
    // Strategy A — N-ary Fusion
    //
    // Merge all operands in the subset into a single FusedContractionNode.
    // No intermediate materialisation; the cost is purely the one-shot kernel.
    // =========================================================================
    static void evaluate_nary_fusion(
        DPState& state,
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& out_indices,
        const std::unordered_map<Index, int>& dim_sizes)
    {
        const FusedCostResult eval = CostModel::evaluate_fused(operands, out_indices, dim_sizes);

        if (state.try_relax(eval.total_cost, make_fused_node(operands, out_indices, eval)))
        {
            // (relaxation succeeded; state updated in-place)
        }
    }

    // =========================================================================
    // Strategy B — Binary Contraction Sequence
    //
    // For every non-trivial partition (left_mask, right_mask) of the current
    // mask into two disjoint non-empty subsets:
    //
    //   total_cost = dp[left].min_cost + dp[right].min_cost + contract(lhs, rhs)
    //
    // The left/right sub-problems are already solved (smaller subsets were
    // processed in earlier DP iterations). We only need to evaluate the cost
    // of contracting the two best-plan outputs together.
    //
    // We deduplicate symmetric partitions (A,B) == (B,A) by enforcing
    // left_mask < right_mask.
    // =========================================================================
    static void evaluate_binary_splits(
        std::vector<DPState>&                          dp,
        int                                            mask,
        const std::vector<Index>&                      out_indices,
        const std::unordered_map<Index, int>&          dim_sizes,
        const std::vector<std::shared_ptr<ExprNode>>& all_inputs,
        const std::vector<Index>&                      global_out_indices)
    {
        // Enumerate all non-empty proper subsets via the bit-manipulation trick
        for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
            const int left_mask  = sub;
            const int right_mask = mask ^ sub;

            // Deduplicate: only process each unordered pair once
            if (left_mask >= right_mask) continue;

            const DPState& left_state  = dp[left_mask];
            const DPState& right_state = dp[right_mask];
            if (!left_state.is_valid() || !right_state.is_valid()) continue;

            // The two sub-problem results become the operands of the final step
            const std::vector<std::shared_ptr<ExprNode>> pair_ops = {
                left_state.best_ast, right_state.best_ast};

            const FusedCostResult pair_eval =
                CostModel::evaluate_fused(pair_ops, out_indices, dim_sizes);

            const double total = left_state.min_cost
                               + right_state.min_cost
                               + pair_eval.total_cost;

            dp[mask].try_relax(total, make_fused_node(pair_ops, out_indices, pair_eval));
        }
    }

    // =========================================================================
    // Strategy C — Pivot Index Decomposition
    //
    // For each CONTRACTED index k in this subset (i.e. k ∈ I(subset) but
    // k ∉ output_indices):
    //
    //   1. Partition operands into:
    //        G_k    = {ops that contain k}          (must be ≥ 2 for this to help)
    //        G_rest = {ops that do NOT contain k}   (must be ≥ 1)
    //
    //   2. Compute intermediate T by fusing G_k.
    //        T's output = indices of G_k that are needed by G_rest or global_out
    //                     (k itself is contracted within this step)
    //
    //   3. Contract T with G_rest to produce the final result for this subset.
    //
    // This strategy decomposes a deep N-ary loop nest into two shallower ones.
    // G_rest never iterates over k, reducing its working set and improving
    // cache locality.  The trade-off is writing T to memory between steps.
    //
    // Example:
    //   A(i,k) B(k,j) C(j,l)  output: (i,l)
    //   Contracted: {k, j}
    //
    //   Pivot k:  T(i,j) = A(i,k) * B(k,j)   ← G_k = {A, B}, contracts k
    //             R(i,l) = T(i,j) * C(j,l)    ← G_rest = {C}
    //   This separates the 3-way loop nest into two 2-way ones.
    // =========================================================================
    static void evaluate_pivot_decomposition(
        std::vector<DPState>&                          dp,
        int                                            mask,
        const std::vector<std::shared_ptr<ExprNode>>& subset_ops,
        const std::vector<Index>&                      out_indices,
        const std::unordered_map<Index, int>&          dim_sizes,
        const std::vector<std::shared_ptr<ExprNode>>& all_inputs,
        const std::vector<Index>&                      global_out_indices)
    {
        // Identify contracted indices for this subset
        const IndexClassification idx_class =
            CostModel::classify_indices(subset_ops, out_indices);

        for (Index pivot : idx_class.contracted_indices) {
            // Partition operands
            std::vector<std::shared_ptr<ExprNode>> g_k, g_rest;
            for (const auto& op : subset_ops) {
                const auto idxs = op->get_indices();
                const bool has_pivot = std::find(idxs.begin(), idxs.end(), pivot)
                                       != idxs.end();
                (has_pivot ? g_k : g_rest).push_back(op);
            }

            // A pivot is only useful if BOTH groups are non-empty and G_k
            // has at least 2 tensors (otherwise it's already binary)
            if (g_k.size() < 2 || g_rest.empty()) continue;

            // Determine what T must output: indices of G_k needed by G_rest
            // or by the global output.  pivot k is NOT included because it is
            // contracted inside G_k.
            //
            // We reuse infer_output_indices with G_k as the subset and
            // subset_ops as "all operands", so "remaining" = G_rest.
            const std::vector<Index> t_out_indices =
                CostModel::infer_output_indices(g_k, subset_ops, global_out_indices);

            // Step 1 cost: fuse G_k → T
            const FusedCostResult step1_eval =
                CostModel::evaluate_fused(g_k, t_out_indices, dim_sizes);
            const auto t_node = make_fused_node(g_k, t_out_indices, step1_eval);

            // Step 2 operands: T + G_rest
            std::vector<std::shared_ptr<ExprNode>> step2_ops = {t_node};
            step2_ops.insert(step2_ops.end(), g_rest.begin(), g_rest.end());

            // Step 2 cost: fuse {T} ∪ G_rest → final output of this subset
            const FusedCostResult step2_eval =
                CostModel::evaluate_fused(step2_ops, out_indices, dim_sizes);

            const double total = step1_eval.total_cost + step2_eval.total_cost;
            // Note: no prior sub-problem costs to add — pivot decomposition is
            // itself a complete plan for this subset (both steps are planned here).

            if (total < dp[mask].min_cost) {
                // Build a two-level AST: outer node contracts T with G_rest
                // The intermediate T node is embedded as an operand
                dp[mask].try_relax(total,
                    make_fused_node(step2_ops, out_indices, step2_eval));
            }
        }
    }

    // =========================================================================
    // AST Construction Helper
    // =========================================================================
    static std::shared_ptr<FusedContractionNode> make_fused_node(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      out_indices,
        const FusedCostResult&                         eval)
    {
        return std::make_shared<FusedContractionNode>(
            operands,
            out_indices,
            eval.estimated_out_nnz,
            eval.loop_order,
            eval.out_format);
    }

    // =========================================================================
    // Post-pass: apply_recommended_formats
    //
    // Walks the final AST produced by the DP and rewrites every TensorNode
    // leaf to use the storage format recommended by its parent kernel's
    // loop_iteration_order. This is what causes:
    //
    //   SpGEMM:  B(i,k) declared in COO  →  B(i,k):[i(C),k(C)]  [CSR]
    //            C(k,j) declared in COO  →  C(j,k):[j(C),k(C)]  [CSC]
    //
    // when printed with to_string().
    //
    // Design: we never mutate shared_ptrs from the DP table (multiple DP
    // states may reference the same node). Instead we copy-on-write: any
    // TensorNode or FusedContractionNode that needs updating gets a fresh
    // allocation. Nodes that are already correct are returned as-is.
    // =========================================================================
    static std::shared_ptr<ExprNode> apply_recommended_formats(
        const std::shared_ptr<ExprNode>& node,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        // Leaf: format will be set when we process it in the context of its
        // parent kernel. Return unchanged here — the parent handles it.
        if (std::dynamic_pointer_cast<TensorNode>(node)) {
            return node;
        }

        auto fused = std::dynamic_pointer_cast<FusedContractionNode>(node);
        if (!fused) return node; // unknown node type; pass through

        // Recursively rewrite each operand, then apply format to TensorNode leaves
        std::vector<std::shared_ptr<ExprNode>> new_operands;
        new_operands.reserve(fused->operands.size());
        bool any_changed = false;

        for (const auto& op : fused->operands) {
            if (auto t = std::dynamic_pointer_cast<TensorNode>(op)) {
                // Recommend the format this leaf needs for the current kernel
                auto rec = FormatSelector::recommend_for(
                    *t, fused->loop_iteration_order, dim_sizes);

                if (FormatSelector::needs_reformat(*t, rec.format.mode_order) ||
                    t->format_label != rec.label)
                {
                    // Copy-on-write: create a new TensorNode with updated format
                    auto new_t = std::make_shared<TensorNode>(*t);
                    new_t->format       = rec.format;
                    new_t->format_label = rec.label;
                    new_t->rationale    = rec.rationale;
                    new_operands.push_back(new_t);
                    any_changed = true;
                } else {
                    new_operands.push_back(op);
                }
            } else {
                // Recurse into sub-trees (intermediates from binary splits
                // or pivot decomposition)
                auto rewritten = apply_recommended_formats(op, dim_sizes);
                if (rewritten != op) any_changed = true;
                new_operands.push_back(rewritten);
            }
        }

        if (!any_changed) return node;

        // Return an updated FusedContractionNode referencing the rewritten operands
        return std::make_shared<FusedContractionNode>(
            std::move(new_operands),
            fused->out_indices,
            fused->cached_nnz,
            fused->loop_iteration_order,
            fused->output_format);
    }

}; // class DPOptimizer