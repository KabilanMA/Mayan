#pragma once
#include "ast.h"
#include <vector>
#include <unordered_set>
#include <string>
#include <sstream>
#include <memory>
#include <cmath>

// =============================================================================
// InputFormatRecommendation
//
// The result of asking "given this loop order, how should this tensor be stored?"
//
//   format    : the recommended StorageFormat (mode_order + level_types)
//   label     : human-readable name  ("CSR", "CSC", "CSF[i,j,k]", ...)
//   rationale : one-sentence explanation of the decision
// =============================================================================
struct InputFormatRecommendation {
    StorageFormat format;
    std::string   label;
    std::string   rationale;
};

// =============================================================================
// FormatSelector
//
// Decides the optimal CSF storage format for each input tensor given the
// loop iteration order chosen by the cost model.
//
// Core rule  (Concordance Principle):
//   For each input tensor T with indices I_T:
//     optimal_mode_order(T) = subsequence of loop_order that is in I_T
//
//   Rationale: the outermost loop variable must be the outermost CSF level
//   so that each fiber tree node is visited exactly once in order (direct
//   tree traversal).  Any other ordering forces either random access or a
//   full linear scan of all fibers.
//
// Label conventions for 2-D tensors (the most common case):
//   - mode_order matches original declaration order  → "CSR"
//     (rows are the outer loop: standard compressed-sparse-row)
//   - mode_order is the reverse of declaration order → "CSC"
//     (columns are the outer loop: compressed-sparse-column)
//
//   For SpGEMM  A(i,j) = B(i,k) * C(k,j), loop order [i, j, k]:
//     B(i,k) → subsequence [i,k] = original → CSR   ✓
//     C(k,j) → subsequence [j,k] ≠ original → CSC   ✓
//
// Level type policy for COO-derived inputs:
//   - All levels COMPRESSED (standard CSF / CSR / CSC).
//   - We never make an INPUT level DENSE unless the tensor's own density
//     along that mode exceeds DENSE_THRESHOLD, because a DENSE level
//     allocates O(dim_size) memory regardless of NNZ, which wastes
//     bandwidth for sparse inputs.
//
// COO → CSF conversion cost:
//   Converting a COO tensor to CSF requires sorting by the new mode order.
//   The cost is modelled as:  NNZ × log(NNZ) × SORT_CONSTANT
//   This is added to the kernel's total cost so the DP can correctly
//   weigh re-ordering against just accepting a discordant access pattern.
// =============================================================================
class FormatSelector {
public:
    static constexpr double DENSE_THRESHOLD = 0.30; // density above which a level is DENSE
    static constexpr double SORT_CONSTANT   = 1e-2; // relative cost of a comparison sort

    // -------------------------------------------------------------------------
    // Recommend a storage format for a single tensor given the kernel loop order.
    // -------------------------------------------------------------------------
    static InputFormatRecommendation recommend_for(
        const TensorNode&          tensor,
        const std::vector<Index>&  loop_order,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        // ---- Build optimal mode order: subsequence of loop_order ∩ tensor.indices
        const std::unordered_set<Index> tensor_idx_set(
            tensor.indices.begin(), tensor.indices.end());

        std::vector<Index> mode_order;
        mode_order.reserve(tensor.indices.size());
        for (Index idx : loop_order) {
            if (tensor_idx_set.count(idx)) {
                mode_order.push_back(idx);
            }
        }

        // Fallback: if loop_order doesn't mention any of this tensor's indices
        // (shouldn't happen in a well-formed plan), keep original declaration order.
        if (mode_order.empty()) {
            mode_order = tensor.indices;
        }

        // ---- Choose level types
        std::vector<LevelType> level_types =
            choose_level_types(tensor, mode_order, dim_sizes);

        StorageFormat fmt{level_types, mode_order};

        // ---- Label the format
        const std::string label    = classify_label(tensor.indices, mode_order);
        const std::string rational = build_rationale(tensor, loop_order, mode_order, label);

        return {std::move(fmt), label, rational};
    }

    // -------------------------------------------------------------------------
    // Batch version: one recommendation per operand.
    // Only processes TensorNode leaves; FusedContractionNode intermediates
    // already carry their output_format from the DP.
    // -------------------------------------------------------------------------
    static std::vector<InputFormatRecommendation> recommend_all(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      loop_order,
        const std::unordered_map<Index, int>&          dim_sizes = {})
    {
        std::vector<InputFormatRecommendation> results;
        results.reserve(operands.size());

        for (const auto& op : operands) {
            if (auto t = std::dynamic_pointer_cast<TensorNode>(op)) {
                results.push_back(recommend_for(*t, loop_order, dim_sizes));
            } else {
                // Intermediate FusedContractionNode: its format is already
                // determined by the DP; emit a pass-through entry.
                auto fused = std::dynamic_pointer_cast<FusedContractionNode>(op);
                const StorageFormat& fmt = fused ? fused->output_format : StorageFormat{};
                std::string lbl = "intermediate (CSF" + mode_order_str(fmt.mode_order) + ")";
                results.push_back({fmt, lbl, "output of prior fused kernel"});
            }
        }
        return results;
    }

    // -------------------------------------------------------------------------
    // Estimate the COO → CSF conversion cost for a tensor.
    // This is the one-time sort cost to build the CSF tree.
    // -------------------------------------------------------------------------
    static double conversion_cost(const TensorNode& tensor) {
        const double nnz = tensor.estimate_nnz();
        if (nnz <= 1.0) return 0.0;
        return nnz * std::log2(nnz) * SORT_CONSTANT;
    }

    // -------------------------------------------------------------------------
    // Public helpers (also used by dp_optimizer when rewriting leaf nodes)
    // -------------------------------------------------------------------------

    // Return true if the recommended format differs from the tensor's current format.
    static bool needs_reformat(
        const TensorNode&         tensor,
        const std::vector<Index>& recommended_mode_order)
    {
        return tensor.format.mode_order != recommended_mode_order;
    }

    // Produce a human-readable label for a format given original and optimized orders.
    static std::string classify_label(
        const std::vector<Index>& original_indices,
        const std::vector<Index>& optimized_mode_order)
    {
        if (original_indices.size() == 2) {
            if (optimized_mode_order == original_indices) {
                // Outer loop = first declared dimension = "row" → CSR
                return "CSR";
            }
            if (optimized_mode_order.size() == 2 &&
                optimized_mode_order[0] == original_indices[1] &&
                optimized_mode_order[1] == original_indices[0]) {
                // Outer loop = second declared dimension = "column" → CSC
                return "CSC";
            }
        }
        // General case: CSF with explicit mode order annotation
        return "CSF" + mode_order_str(optimized_mode_order);
    }

private:
    // Choose DENSE vs COMPRESSED for each level of the recommended mode order.
    static std::vector<LevelType> choose_level_types(
        const TensorNode&             tensor,
        const std::vector<Index>&     mode_order,
        const std::unordered_map<Index, int>& dim_sizes)
    {
        std::vector<LevelType> level_types;
        level_types.reserve(mode_order.size());

        // Compute per-mode density using nnz_along_dim if available
        for (size_t i = 0; i < mode_order.size(); ++i) {
            Index idx = mode_order[i];

            // Is there fine-grained NNZ-per-slice data for this mode?
            const auto& nnz_map = tensor.metadata.nnz_along_dim;
            auto nnz_it = nnz_map.find(idx);
            bool use_dense = false;

            if (nnz_it != nnz_map.end() && !nnz_it->second.empty()) {
                // Compute average fill-in along this mode
                const auto& slices = nnz_it->second;
                double total = 0.0;
                for (double v : slices) total += v;
                const double avg_nnz_per_slice = total / static_cast<double>(slices.size());

                auto dim_it = dim_sizes.find(idx);
                const double mode_dim = (dim_it != dim_sizes.end())
                    ? static_cast<double>(dim_it->second)
                    : 1.0;
                const double mode_density = avg_nnz_per_slice / mode_dim;
                use_dense = (mode_density > DENSE_THRESHOLD);
            } else {
                // Fallback: use global density for the innermost level only
                use_dense = (i == mode_order.size() - 1) &&
                            (tensor.metadata.global_density > DENSE_THRESHOLD);
            }

            level_types.push_back(use_dense ? LevelType::DENSE : LevelType::COMPRESSED);
        }

        return level_types;
    }

    static std::string mode_order_str(const std::vector<Index>& mode_order) {
        std::string s = "[";
        for (size_t i = 0; i < mode_order.size(); ++i) {
            s += mode_order[i];
            if (i < mode_order.size() - 1) s += ",";
        }
        return s + "]";
    }

    static std::string build_rationale(
        const TensorNode&         tensor,
        const std::vector<Index>& loop_order,
        const std::vector<Index>& mode_order,
        const std::string&        label)
    {
        std::ostringstream oss;
        oss << tensor.name << " stored as " << label
            << ": loop order [";
        for (size_t i = 0; i < loop_order.size(); ++i) {
            oss << loop_order[i];
            if (i < loop_order.size() - 1) oss << ",";
        }
        oss << "] → outer mode is '" << mode_order[0] << "'";
        if (tensor.format.mode_order != mode_order) {
            oss << " (requires COO re-sort from original ["
                << mode_order_str(tensor.format.mode_order) << "])";
        } else {
            oss << " (concordant with original layout, no re-sort needed)";
        }
        return oss.str();
    }
};