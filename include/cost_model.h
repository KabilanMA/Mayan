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

#include "util.h"


// FREE: appears in the output; forms the outer loop nest
// CONTRACTED: summed over; forms the inner loop nest; not in the output
// classifies each loop index for this kernel
enum class IndexRole { FREE, CONTRACTED };

// Classifies loop indices for a kernel into three distinct roles.
struct IndexClassification {
    // Indices that appear in the kernel's output tensor. These form the outer
    // loops of the computation.
    std::vector<Index> free_indices;

    // Indices that are contracted (summed over) and appear in two or more
    // input tensors. These represent the "join" dimensions that connect operands.
    std::vector<Index> shared_contracted_indices;

    // Indices that are contracted but appear in only a single operand. These
    // correspond to internal self-reductions within a tensor before any
    // inter-tensor operations occur. For example, in A(i,j) * B(j,k), 'k' is a
    // private contracted index for B. The effective NNZ of B for the join is
    // based on the number of unique 'j' values, not the total NNZ of B.
    std::vector<Index> private_contracted_indices;

    // A combined list of all contracted indices (both shared and private).
    // These indices are eliminated by the kernel's computation.
    std::vector<Index> contracted_indices;

    // // All unique indices involved in the kernel.
    // std::vector<Index> all_indices;
};

// Records the estimation strategy used by `estimate_output_nnz`.
// This is stored in `KernelCostBreakdown` for logging and analysis.
enum class NnzEstimationMode {
    // The kernel is a pure outer product (no contracted indices).
    // The NNZ is estimated as the product of the NNZs of all operands.
    // Formula: Π NNZ(operand_i)
    OUTER_PRODUCT,

    // The kernel is a broadcast join (shared indices but none are contracted).
    // The NNZ is estimated using KMV on shared indices and fan-out for unshared ones.
    BROADCAST_JOIN,

    // The kernel's contraction graph is a single connected component.
    // The NNZ is estimated using KMV intersection across all operands.
    KMV_INTERSECTION_SINGLE_COMPONENT,

    // The kernel's contraction graph has multiple connected components.
    // KMV intersection is used within each component, and the results are
    // multiplied (Cartesian product) across components.
    KMV_INTERSECTION_MULTI_COMPONENT,

    // Sketch-based estimation was not possible (e.g., missing sketches).
    // The fallback heuristic was used: density * output_volume.
    DENSITY_PRODUCT_FALLBACK,
};

// Contains a detailed breakdown of the costs for a kernel, returned to the DP optimizer.
struct KernelCostBreakdown {
    // Estimated floating-point operations, modeled as the sum of operand NNZs
    // scaled by a divergence factor.
    double compute_cost      = 0.0;
    // Penalty for memory access patterns that are not concordant with the
    // physical storage layout (e.g., CSC access in a CSR-ordered loop).
    double access_penalty    = 0.0;
    // Cost of writing the materialized output tensor to high-bandwidth memory.
    double memory_write_cost = 0.0;
    // The total estimated cost of the kernel, used for DP decisions.
    double total_cost        = 0.0;
    // The method used to estimate the output NNZ.
    NnzEstimationMode nnz_mode = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
};

// Encapsulates all information the DP optimizer needs about a proposed kernel fusion.
struct FusedCostResult {
    // The total cost of this fusion, including compute, memory, and reformatting.
    double total_cost;
    // The estimated number of non-zero elements in the output tensor.
    double estimated_out_nnz;
    // The determined loop order for the kernel (free indices first, then contracted).
    std::vector<Index> loop_order;
    // The recommended storage format for the output tensor.
    StorageFormat out_format;
    // A detailed breakdown of the cost components for debugging and analysis.
    KernelCostBreakdown breakdown;

    // Recommended input formats for each operand. For tensor leaves, this includes
    // any reformatting cost, which is incorporated into `total_cost`.
    std::vector<InputFormatRecommendation> input_formats;

    // Approximated sketch metadata for the intermediate tensor produced by this
    // fusion. This is crucial for enabling multi-level sketch-based estimation.
    SparseMetadata metadata;
};

/**
 * @class CostModel
 * @brief Analyzes and scores proposed tensor contraction kernels.
 *
 * This class provides the core logic for the dynamic programming optimizer.
 * Its main responsibilities are:
 *   1. Index Classification: Categorizing indices as free, shared contracted,
 *      or private contracted.
 *   2. NNZ Estimation: Estimating the cardinality of intermediate tensors
 *      using a hybrid sketch-based model (KMV for intersections, HLL for
 *      projections) or falling back to heuristics.
 *   3. Kernel Costing: Calculating a total cost for a proposed kernel based on
 *      compute, memory access patterns, and data reformatting.
 */
class CostModel {
public:

    /**
     * @brief Evaluates the total cost of fusing a set of operands into a single kernel.
     *
     * This is the main entry point for the cost model. It orchestrates the
     * full analysis of a proposed fusion: index classification, NNZ estimation,
     * loop order determination, format selection, and final cost calculation.
     *
     * @param operands The expression nodes (tensors or intermediates) to be fused.
     * @param output_indices The set of indices that must be present in the output.
     * @param dim_sizes A map from an index to its dimension size, used for
     *                  accurate sparsity and volume calculations.
     * @return A `FusedCostResult` containing the total cost and detailed
     *         breakdown for the proposed fusion.
     */
    static FusedCostResult evaluate_fused(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices,
        const std::unordered_map<Index, int>& dim_sizes = {})
    {
        if (operands.empty()) {
            return {0.0, 0.0, {}, {}, {}, {}};
        }

        // Classify indices as FREE (output) or CONTRACTED (summed over).
        const IndexClassification idx_class = classify_indices(operands, output_indices);

        // Estimate the number of non-zero elements in the output tensor.
        NnzEstimationMode nnz_mode = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
        const double estimated_out_nnz = estimate_output_nnz(operands, output_indices, idx_class, dim_sizes, nnz_mode);

        // Determine the optimal loop order for the kernel.
        const std::vector<Index> loop_order = determine_loop_order(operands, idx_class);

        // Choose the storage format for the output tensor.
        const StorageFormat out_format = choose_output_format(loop_order, output_indices, estimated_out_nnz, dim_sizes);

        // Score the kernel based on compute, access, and write costs.
        KernelCostBreakdown breakdown = score_kernel(operands, loop_order, estimated_out_nnz);
        breakdown.nnz_mode = nnz_mode;

        //    Recommend input formats and account for any necessary re-sorting.
        //    The cost of converting tensors from COO to the recommended CSF
        //    format is added to the total, ensuring the DP optimizer can make
        //    fully informed decisions.
        std::vector<InputFormatRecommendation> input_formats = FormatSelector::recommend_all(operands, loop_order, dim_sizes);

        double coo_conversion_cost = 0.0;
        for (size_t i = 0; i < operands.size(); ++i) {
            if (auto t = get_underlying_tensor(operands[i])) {
                if (FormatSelector::needs_reformat(*t, input_formats[i].format.mode_order)) {
                    coo_conversion_cost += FormatSelector::conversion_cost(*t);
                }
            }
        }

        const double total_cost = breakdown.total_cost + coo_conversion_cost;

        // --- Sketch Propagation for Intermediate ---
        // For the DP optimizer to work across multiple levels, we must provide
        // an estimated sketch for the intermediate tensor this fusion produces.
        //
        // NOTE: The previous naive propagation was flawed, as pointed out.
        // A correct implementation needs to synthesize a new sketch based on
        // the join, which is a complex problem. For now, we don't propagate
        // sketches for intermediates. This limits the DP search depth but
        // prevents incorrect estimations.
        SparseMetadata intermediate_metadata;

        return {total_cost, estimated_out_nnz, loop_order, out_format, breakdown, std::move(input_formats), std::move(intermediate_metadata)};
    }

    /**
     * @brief Infers which indices of a subset of operands must appear in its output.
     *
     * This function determines the required output indices for an intermediate
     * tensor produced by contracting a subset of the total operands.
     *
     * An index must be preserved in the output of a subset `S` if it is
     * needed by any operand *not* in `S`, or if it is part of the final
     * global output of the entire expression.
     *
     * @param subset_operands The subset of operands being considered.
     * @param all_operands All operands in the full expression, used to find
     *                     indices needed by later stages.
     * @param global_out_indices The final output indices of the entire expression.
     * @return A sorted vector of indices that must be output by the subset's kernel.
     */
    static std::vector<Index> infer_output_indices(
        const std::vector<std::shared_ptr<ExprNode>>& subset_operands,
        const std::vector<std::shared_ptr<ExprNode>>& all_operands,
        const std::vector<Index>& global_out_indices)
    {
        std::unordered_set<Index> subset_idx;
        std::unordered_set<Index> remaining_idx;
        const std::unordered_set<Index> global_out(global_out_indices.begin(), global_out_indices.end());

        // Collect all indices appearing within the current subset.
        for (const auto& op : subset_operands) {
            for (Index idx : op->get_indices()) {
                subset_idx.insert(idx);
            }
        }

        // Collect all indices appearing in the *other* operands.
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

        // An index must be output if it's in the subset AND it's either
        // needed by a remaining operand or required in the global output.
        std::vector<Index> output;
        output.reserve(subset_idx.size());
        for (Index idx : subset_idx) {
            if (remaining_idx.count(idx) || global_out.count(idx)) {
                output.push_back(idx);
            }
        }

        // Sort for canonical representation, ensuring stable DP memoization keys.
        std::sort(output.begin(), output.end());
        return output;
    }

    // DONE
    /**
     * @brief Classifies all indices in a set of operands as FREE, SHARED CONTRACTED,
     *        or PRIVATE CONTRACTED.
     *
     * @param operands The tensor operands for the kernel.
     * @param output_indices The indices that must appear in the kernel's output.
     * @return An `IndexClassification` struct containing sorted lists of indices
     *         for each category.
     */
    static IndexClassification classify_indices(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices)
    {
        const std::unordered_set<Index> out_set(output_indices.begin(), output_indices.end());

        // Count the number of operands each index appears in.
        std::unordered_map<Index, int> index_operand_count;
        for (const auto& op : operands) {
            std::unordered_set<Index> seen;
            for (Index idx : op->get_indices()) {
                if (seen.insert(idx).second) {
                    index_operand_count[idx]++;
                }
            }
        }

        IndexClassification result;
        for (const auto& [idx, count] : index_operand_count) {
            if (out_set.count(idx)) {
                result.free_indices.push_back(idx);
            } else {
                result.contracted_indices.push_back(idx);
                if (count >= 2) {
                    result.shared_contracted_indices.push_back(idx);
                } else {
                    result.private_contracted_indices.push_back(idx);
                }
            }
        }

        // Sort all index lists for canonical representation and reproducibility.
        std::sort(result.free_indices.begin(), result.free_indices.end());
        std::sort(result.contracted_indices.begin(), result.contracted_indices.end());
        std::sort(result.shared_contracted_indices.begin(), result.shared_contracted_indices.end());
        std::sort(result.private_contracted_indices.begin(), result.private_contracted_indices.end());
        return result;
    }

private:
    // =========================================================================
    // NNZ Estimation
    // =========================================================================

    /**
     * @brief Estimates the number of non-zero elements in the output of a kernel.
     *
     * This function orchestrates the NNZ estimation by deciding which strategy
     * to use. It prioritizes sketch-based estimation, which uses a hybrid model:
     *   - KMV sketches for intersections (joins and element-wise operations).
     *   - HLL sketches for projections (cardinality of single dimensions).
     *
     * If any operand lacks the necessary sketches, it falls back to a
     * heuristic based on the product of operand densities and the output volume.
     *
     * @param mode_out [out] The estimation mode that was used is stored here.
     * @return The estimated number of non-zero elements.
     */
    static double estimate_output_nnz(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& output_indices,
        const IndexClassification& idx_class,
        const std::unordered_map<Index, int>& dim_sizes,
        NnzEstimationMode& mode_out)
    {
        // For a single operand, NNZ is its projected cardinality after any
        // self-reductions are applied.
        if (operands.size() == 1) {
            const std::unordered_set<Index> private_set(idx_class.private_contracted_indices.begin(), idx_class.private_contracted_indices.end());
            return projected_nnz(*operands[0], private_set, dim_sizes);
        }

        // If sketches are available, use the advanced sketch-based model.
        if (any_operand_has_sketch(operands)) {
            return estimate_nnz_via_sketch(operands, idx_class, dim_sizes, mode_out);
        }
        
        // Otherwise, fall back to the density-product heuristic.
        mode_out = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
        return estimate_nnz_density_product(operands, output_indices, dim_sizes);
    }

    /// @brief Checks if any operand in the list has any type of sketch data.
    static bool any_operand_has_sketch(const std::vector<std::shared_ptr<ExprNode>>& operands)
    {
        for (const auto& op : operands) {
            if (!op->metadata.hll_sketch.empty() || !op->metadata.kmv_sketch.empty() ||
                !op->metadata.mode_sketches.empty() || !op->metadata.mode_kmv_sketches.empty())
            {
                return true;
            }
        }
        return false;
    }

    // DONE
    /// @brief Checks if the operation is purely element-wise across all operands.
    /// This is true if all operands share the exact same set of indices.
    static bool is_element_wise(const std::vector<std::shared_ptr<ExprNode>>& operands)
    {
        if (operands.size() <= 1) 
            return true;

        const auto& first_indices = operands[0]->get_indices();
        const std::unordered_set<Index> base_indices(first_indices.begin(), first_indices.end());

        for (size_t i = 1; i < operands.size(); ++i) {
            const auto& current_indices = operands[i]->get_indices();
            if (current_indices.size() != base_indices.size()) 
                return false;
            const std::unordered_set<Index> curr_set(current_indices.begin(), current_indices.end());
            if (base_indices != curr_set) 
                return false;
        }
        return true;
    }

    /**
     * @brief Implements the sketch-based NNZ estimation model.
     *
     * This model handles three cases:
     *  1. Element-wise operations: Prefers KMV for intersection. Falls back to
     *     the minimum operand NNZ as a safe upper bound.
     *  2. Outer products: No contracted indices. Result is the product of
     *     operand NNZs.
     *  3. Contractions:
     *     a. Builds a contraction graph to identify connected components.
     *     b. Computes the "effective NNZ" of each operand by projecting out
     *        any private contracted indices using `projected_nnz`.
     *     c. For each component, estimates NNZ using `estimate_component_nnz`,
     *        which prioritizes KMV intersection on shared indices.
     *     d. The final NNZ is the product of the NNZ estimates from each
     *        independent component.
     */
    static double estimate_nnz_via_sketch(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const IndexClassification& idx_class,
        const std::unordered_map<Index, int>& dim_sizes,
        NnzEstimationMode& mode_out)
    {
        const size_t N = operands.size();
        assert(N > 1);

        // --- Case 1: Element-wise Operations ---
        // For element-wise operations, KMV is strongly preferred for its
        // accuracy in multi-way intersections.

        // currently I am thinking of supporting only Hadamard product, C(i,j)=A(i,j) * B(i,j). 
        // This acts as a logical AND filter where KMV is better. 
        // If in future, planning to support "Addition" between two tensors, 
        // which will become a OR filter where one NN can produce a NN in the output (Then HLL for that too).
        if (is_element_wise(operands)) {
            // Try to use full-tensor KMV sketches if all operands have them.
            // mayan_debug(operands);
            bool has_all_full_kmv = true;
            std::vector<KMinValues> kmvs;
            kmvs.reserve(N);
            for (const auto& op : operands) {
                if (op->metadata.kmv_sketch.empty()) {
                    has_all_full_kmv = false;
                    break;
                }
                kmvs.emplace_back(op->metadata.kmv_sketch);
            }

            if (has_all_full_kmv && !kmvs.empty()) {
                std::vector<const KMinValues*> ptrs;
                ptrs.reserve(kmvs.size());
                for (const auto& k : kmvs) 
                    ptrs.push_back(&k);
                
                mode_out = NnzEstimationMode::KMV_INTERSECTION_SINGLE_COMPONENT;
                return std::max(1.0, KMinValues::estimate_intersection(ptrs));
            }

            // Fallback for element-wise: The output NNZ is bounded by the
            // smallest input NNZ.
            // hopefully any generator using this IR would set the KMV for element-wise properly so that it wouldn't fall here.
            mode_out = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
            double min_nnz = std::numeric_limits<double>::max();
            for (const auto& op : operands) {
                min_nnz = std::min(min_nnz, op->estimate_nnz());
            }
            return std::max(1.0, min_nnz);
        }

        // --- Case 2: Broadcast / Partial Join OR Outer Product ---
        if (idx_class.contracted_indices.empty()) {
            std::unordered_map<Index, int> index_counts;
            std::vector<Index> all_indices_vec;
            for(const auto& op : operands) {
                for(Index idx : op->get_indices()) {
                    if(index_counts.find(idx) == index_counts.end()) {
                        index_counts[idx] = 0;
                        all_indices_vec.push_back(idx);
                    }
                    index_counts[idx]++;
                }
            }

            std::vector<Index> shared_indices;
            for(Index idx : all_indices_vec) {
                if(index_counts[idx] > 1) {
                    shared_indices.push_back(idx);
                }
            }

            if (shared_indices.empty()) {
                // --- Pure Outer Product ---
                mode_out = NnzEstimationMode::OUTER_PRODUCT;
                double product = 1.0;
                for (const auto& op : operands)
                    product *= op->estimate_nnz();
                return std::max(1.0, product);

            } else {
                // --- Broadcast / Partial Join ---
                mode_out = NnzEstimationMode::BROADCAST_JOIN;
                double estimate = -1.0;

                for (Index shared_idx : shared_indices) {
                    std::vector<std::shared_ptr<const ExprNode>> ops_with_shared_idx;
                    bool all_have_kmv = true;

                    for (const auto& op : operands) {
                        bool has_idx = false;
                        for(Index i : op->get_indices()) if(i == shared_idx) has_idx = true;

                        if (has_idx) {
                            if (op->metadata.mode_kmv_sketches.count(shared_idx)) {
                                ops_with_shared_idx.push_back(op);
                            } else {
                                all_have_kmv = false;
                                break;
                            }
                        }
                    }
                    if (!all_have_kmv) continue;

                    std::vector<KMinValues> kmvs;
                    for(const auto& op : ops_with_shared_idx) {
                        kmvs.emplace_back(op->metadata.mode_kmv_sketches.at(shared_idx));
                    }
                    std::vector<const KMinValues*> ptrs;
                    for(const auto& k : kmvs) ptrs.push_back(&k);

                    double intersection_size = KMinValues::estimate_intersection(ptrs);
                    double fanout_product = 1.0;

                    for(const auto& op : operands) {
                        bool has_idx = false;
                        for(Index i : op->get_indices()) if(i == shared_idx) has_idx = true;

                        if (has_idx) {
                            const KMinValues kmv(op->metadata.mode_kmv_sketches.at(shared_idx));
                            double card_j = kmv.estimate();
                            if (card_j > 0) {
                                fanout_product *= op->estimate_nnz() / card_j;
                            }
                        } else {
                            fanout_product *= op->estimate_nnz();
                        }
                    }
                    estimate = intersection_size * fanout_product;
                    break; 
                }

                if (estimate > 0) {
                    return std::max(1.0, estimate);
                }

                // Fallback for broadcast join
                double product = 1.0;
                for (const auto& op : operands) product *= op->estimate_nnz();
                return std::max(1.0, product);
            }
        }

        // --- Case 3: Contractions ---
        // Identify private and shared contracted indices for each operand.
        const std::unordered_set<Index> private_set(idx_class.private_contracted_indices.begin(), idx_class.private_contracted_indices.end());
        const std::unordered_set<Index> contracted_set(idx_class.contracted_indices.begin(), idx_class.contracted_indices.end());

        std::vector<std::unordered_set<Index>> op_private(N);
        std::vector<std::unordered_set<Index>> op_contracted(N);
        for (size_t i = 0; i < N; ++i) {
            for (Index idx : operands[i]->get_indices()) {
                if (private_set.count(idx)) op_private[i].insert(idx);
                if (contracted_set.count(idx)) op_contracted[i].insert(idx);
            }
        }

        // Build a contraction graph using a union-find data structure.
        // An edge exists between two operands if they share a contracted index.
        std::vector<int> parent(N);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<int(int)> find = 
            [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
        auto unite = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a != b) parent[a] = b;
        };

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                for (Index idx : op_contracted[i]) {
                    // Edges are only formed by SHARED contracted indices.
                    if (!private_set.count(idx) && op_contracted[j].count(idx)) {
                        unite(i, j);
                        break;
                    }
                }
            }
        }

        // Group operands into connected components.
        std::unordered_map<int, std::vector<int>> components;
        for (size_t i = 0; i < N; ++i) {
            components[find(i)].push_back(i);
        }

        // Compute the "effective" NNZ for each operand, which is its cardinality
        // after projecting away any private contracted indices.
        std::vector<double> effective_nnz(N);
        for (size_t i = 0; i < N; ++i) {
            effective_nnz[i] = projected_nnz(*operands[i], op_private[i], dim_sizes);
        }

        // The total NNZ is determined by how the independent components combine.
        // Components that share free indices are "element-wise" multiplied (intersected).
        // Components that are disjoint on free indices form a Cartesian product.
        std::vector<double> component_nnzs;
        std::vector<std::unordered_set<Index>> component_free_indices_list;

        for (const auto& [root, members] : components) {
            double component_nnz;
            if (members.size() == 1) {
                component_nnz = effective_nnz[members[0]];
            } else {
                component_nnz = estimate_component_nnz(
                    operands, members, op_contracted, private_set,
                    effective_nnz, dim_sizes);
            }
            component_nnzs.push_back(component_nnz);

            std::unordered_set<Index> component_free_indices;
            for(int member_idx : members) {
                for(Index idx : operands[member_idx]->get_indices()) {
                    if (contracted_set.find(idx) == contracted_set.end()) {
                        component_free_indices.insert(idx);
                    }
                }
            }
            component_free_indices_list.push_back(component_free_indices);
        }

        if (component_nnzs.empty()) {
            return 1.0;
        }

        // Group components that share free indices using another union-find.
        std::vector<int> component_parent(component_nnzs.size());
        std::iota(component_parent.begin(), component_parent.end(), 0);
        std::function<int(int)> find_comp = 
            [&](int x) { return component_parent[x] == x ? x : component_parent[x] = find_comp(component_parent[x]); };
        auto unite_comp = [&](int a, int b) {
            a = find_comp(a); b = find_comp(b);
            if (a != b) component_parent[a] = b;
        };

        for (size_t i = 0; i < component_free_indices_list.size(); ++i) {
            for (size_t j = i + 1; j < component_free_indices_list.size(); ++j) {
                for (Index idx : component_free_indices_list[i]) {
                    if (component_free_indices_list[j].count(idx)) {
                        unite_comp(i, j);
                        break;
                    }
                }
            }
        }
        
        std::unordered_map<int, std::vector<int>> free_groups;
        for(size_t i=0; i<component_nnzs.size(); ++i) {
            free_groups[find_comp(i)].push_back(static_cast<int>(i));
        }

        double total_nnz = 1.0;
        for (const auto& [root, group_members] : free_groups) {
            // For components in a group (sharing free indices), estimate intersection with min().
            double group_nnz = component_nnzs[group_members[0]];
            for (size_t i = 1; i < group_members.size(); ++i) {
                group_nnz = std::min(group_nnz, component_nnzs[group_members[i]]);
            }
            // Multiply the results from independent groups (Cartesian product).
            total_nnz *= group_nnz;
        }

        mode_out = (components.size() == 1)
            ? NnzEstimationMode::KMV_INTERSECTION_SINGLE_COMPONENT
            : NnzEstimationMode::KMV_INTERSECTION_MULTI_COMPONENT;

        return std::max(1.0, total_nnz);
    }

    /**
     * @brief Estimates the NNZ of a tensor after projecting out its private indices.
     *
     * This is crucial for handling self-reductions. For an operation like
     * `A(i,j) * B(j,k)`, where `k` is a private index of `B`, the effective
     * contribution of `B` to the join is not its total NNZ, but the number of
     * unique `j` indices it contains. This function estimates that projected
     * cardinality.
     *
     * It uses a cascading series of methods, from most to least accurate:
     *   1. Per-Mode HLL Sketches: If sketches exist for all surviving dimensions,
     *      they are used to get a highly accurate projection.
     *   2. Full HLL Sketch: The HLL sketch of the entire tensor provides an
     *      upper bound on the projected cardinality.
     *   3. Birthday Problem Formula: A statistical estimate based on the
     *      tensor's total NNZ and the volume of the surviving dimensions.
     *   4. Density Fallback: A simple heuristic that assumes uniform density
     *      across the private dimensions.
     *
     * @return The estimated NNZ after projection.
     */
    static double projected_nnz(
        const ExprNode& op,
        const std::unordered_set<Index>& private_indices,
        const std::unordered_map<Index, int>& dim_sizes)
    {
        if (private_indices.empty()) {
            // No self-reduction; effective NNZ is the total NNZ.
            return op.estimate_nnz();
        }

        // Identify the indices that are NOT being projected out.
        std::vector<Index> surviving;
        for (Index idx : op.get_indices()) {
            if (!private_indices.count(idx)) 
                surviving.push_back(idx);
        }

        // --- Method 1: Per-Mode HLL Sketches (Most Accurate) ---
        // If a sketch was created for each surviving dimension, we can estimate
        // the cardinality of the projected space.
        bool all_mode_sketches = !surviving.empty();
        for (Index idx : surviving) {
            if (op.metadata.mode_sketches.find(idx) == op.metadata.mode_sketches.end()) {
                all_mode_sketches = false;
                break;
            }
        }

        if (all_mode_sketches) {
            if (surviving.size() == 1) {
                // For a single surviving index, its per-mode sketch gives the direct answer.
                HyperLogLog hll(op.metadata.mode_sketches.at(surviving[0]));
                return std::max(1.0, hll.estimate());
            } else {
                // For multiple surviving indices, HLL intersection is invalid.
                // Instead, multiply the marginal cardinalities (from each mode
                // sketch) and cap the result by the total tensor NNZ.
                double marginal_product = 1.0;
                for (Index idx : surviving) {
                    HyperLogLog hll(op.metadata.mode_sketches.at(idx));
                    marginal_product *= hll.estimate();
                }
                return std::max(1.0, std::min(op.estimate_nnz(), marginal_product));
            }
        }

        // --- Method 2: Full HLL Sketch ---
        // The full tensor's HLL sketch counts unique tuples. This is a good
        // upper bound on the projected cardinality.
        if (!op.metadata.hll_sketch.empty()) {
            HyperLogLog hll(op.metadata.hll_sketch);
            return std::max(1.0, hll.estimate());
        }

        // --- Method 3: Birthday Problem Formula ---
        // E[unique] = D * (1 - exp(-NNZ / D)), where D is the volume of the
        // space of surviving dimensions.
        double surviving_volume = 1.0;
        bool have_all_dims = !surviving.empty();
        for (Index idx : surviving) {
            auto it = dim_sizes.find(idx);
            if (it == dim_sizes.end()) {
                have_all_dims = false; 
                break;
            }
            surviving_volume *= static_cast<double>(it->second);
        }

        if (have_all_dims && surviving_volume > 0.0) {
            const double raw_nnz = op.estimate_nnz();
            const double E = surviving_volume * (1.0 - std::exp(-raw_nnz / surviving_volume));
            return std::max(1.0, E);
        }

        // --- Method 4: Density Fallback (Least Accurate) ---
        // Assume uniform density and divide total NNZ by the volume of the
        // dimensions being projected out.
        double private_volume = 1.0;
        for (Index idx : private_indices) {
            auto it = dim_sizes.find(idx);
            private_volume *= (it != dim_sizes.end())
                ? static_cast<double>(it->second)
                : std::max(1.0, std::sqrt(op.estimate_nnz())); // Rough fallback
        }
        return std::max(1.0, op.estimate_nnz() / private_volume);
    }

    /**
     * @brief Estimates the NNZ for a single connected component in the contraction graph.
     *
     * This function handles the intersection of multiple operands that are
     * connected by one or more shared contracted indices.
     *
     * The preferred strategy is to use per-mode KMV sketches. If all operands
     * in the component have a KMV sketch for a given shared index, a multi-way
     * intersection is performed on that index. This is highly accurate.
     *
     * If per-mode KMV fails, it falls back to an AGM-based estimator, which
     * provides a sound upper bound.
     *
     * @return The estimated NNZ of the component's output.
     */
    static double estimate_component_nnz(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<int>&                        members,
        const std::vector<std::unordered_set<Index>>&  op_contracted,
        const std::unordered_set<Index>&               private_set,
        const std::vector<double>&                     effective_nnz,
        const std::unordered_map<Index, int>&          dim_sizes)
    {
        // Identify all indices that are shared and contracted within this component.
        std::unordered_set<Index> shared_in_component;
        for (int i : members) {
            for (Index idx : op_contracted[i]) {
                if (!private_set.count(idx)) shared_in_component.insert(idx);
            }
        }

        // --- Strategy 1: Per-Mode KMV Intersection ---
        // Try to find a shared index for which all component members have a KMV sketch.
        for (Index shared_idx : shared_in_component) {
            bool all_have_kmv_sketch = true;
            for (int m : members) {
                if (operands[m]->metadata.mode_kmv_sketches.find(shared_idx) ==
                    operands[m]->metadata.mode_kmv_sketches.end()) {
                    all_have_kmv_sketch = false;
                    break;
                }
            }

            if (all_have_kmv_sketch) {
                std::vector<KMinValues> kmvs;
                kmvs.reserve(members.size());
                for (int m : members) {
                    kmvs.emplace_back(operands[m]->metadata.mode_kmv_sketches.at(shared_idx));
                }
                std::vector<const KMinValues*> ptrs;
                for (const auto& k : kmvs) ptrs.push_back(&k);

                // Estimate the number of matching coordinates on the shared index.
                const double matching_k = KMinValues::estimate_intersection(ptrs);
                
                // Estimate the number of output tuples for each match. Instead of a
                // simple product of fanouts (which assumes independence and leads to
                // overestimation), use the geometric mean to produce a more
                // conservative estimate.
                double tuples_per_match = 1.0;
                if (!members.empty()) {
                    double fanout_product = 1.0;
                    for (int m : members) {
                        KMinValues kmv_m(operands[m]->metadata.mode_kmv_sketches.at(shared_idx));
                        const double unique_k_m = std::max(1.0, kmv_m.estimate());
                        fanout_product *= std::max(1.0, effective_nnz[m] / unique_k_m);
                    }
                    tuples_per_match = std::pow(fanout_product, 1.0 / members.size());
                }

                return std::max(1.0, matching_k * tuples_per_match);
            }
        }

        // --- Strategy 2: AGM Bound Fallback ---
        return estimate_nnz_via_agm(operands, members, effective_nnz, shared_in_component, dim_sizes);
    }
    
    /**
     * @brief Estimates join size using the AGM-Wannier bound.
     *
     * This provides a worst-case upper bound on the join size, which is tighter
     * than simple min-cardinality. It's used as a fallback when fine-grained
     * per-mode KMV sketches are not available for a multi-way join.
     *
     * The bound is computed as:
     *   `Bound = product(NNZ_i) / product(dom(j))`
     * for i in operands and j in shared indices, where `dom(j)` is the
     * domain size of the shared index `j`.
     *
     * @return The estimated NNZ based on the AGM bound.
     */
    static double estimate_nnz_via_agm(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<int>&                        members,
        const std::vector<double>&                     effective_nnz,
        const std::unordered_set<Index>&               shared_in_component,
        const std::unordered_map<Index, int>&          dim_sizes)
    {
        double numerator = 1.0;
        for (int m : members) {
            numerator *= effective_nnz[m];
        }

        double denominator = 1.0;
        for (Index idx : shared_in_component) {
            auto it = dim_sizes.find(idx);
            if (it != dim_sizes.end()) {
                denominator *= static_cast<double>(it->second);
            } else {
                // If dim size is unknown, fallback to a simple min-cardinality
                // as the AGM bound cannot be computed.
                double min_nnz = std::numeric_limits<double>::max();
                for (int m : members) min_nnz = std::min(min_nnz, effective_nnz[m]);
                return std::max(1.0, min_nnz);
            }
        }

        // To prevent division by zero, and also because a join can't be smaller
        // than the largest input (in some cases), we take the min of the AGM bound
        // and the max effective nnz. More sophisticated bounds exist, but this is a
        // reasonable heuristic.
        double max_effective_nnz = 0.0;
        for (int m : members) {
            max_effective_nnz = std::max(max_effective_nnz, effective_nnz[m]);
        }

        if (denominator > 1.0) {
            return std::max(max_effective_nnz, numerator / denominator);
        }

        return std::max(1.0, numerator);
    }


    /**
     * @brief Fallback NNZ estimation using a density-product heuristic.
     *
     * This method is used when sketch-based estimation is not possible.
     * It estimates the output density as the product of the input densities
     * and multiplies this by the volume of the output tensor's space.
     *
     *   `output_density ≈ Π density(operand_i)`
     *   `output_volume = Π size(free_index_j)`
     *   `estimated_nnz = output_density * output_volume`
     *
     * @return The estimated number of non-zero elements.
     */
    static double estimate_nnz_density_product(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      output_indices,
        const std::unordered_map<Index, int>&          dim_sizes)
    {
        // The combined density is the product of individual operand densities.
        double density_product = 1.0;
        for (const auto& op : operands) {
            density_product *= op->metadata.global_density;
        }
        density_product = std::max(density_product, 1e-9); // Clamp to avoid underflow.

        // The output volume is the product of the sizes of the output dimensions.
        double out_volume = 1.0;
        for (Index idx : output_indices) {
            auto it = dim_sizes.find(idx);
            if (it != dim_sizes.end()) {
                out_volume *= static_cast<double>(it->second);
            } else {
                // If a dimension size is unknown, use a rough heuristic based
                // on the geometric mean of operand NNZs.
                double max_nnz = 0.0;
                for (const auto& op : operands) {
                    max_nnz = std::max(max_nnz, op->estimate_nnz());
                }
                out_volume *= std::sqrt(max_nnz);
            }
        }

        return std::max(1.0, density_product * out_volume);
    }

    // =========================================================================
    // Loop Order Determination
    // =========================================================================

    /**
     * @brief Determines the kernel's loop order using a two-phase,
     *        NNZ-weighted concordance sort.
     *
     * The goal is to find a loop order that maximizes concordant memory access
     * across all input tensors, minimizing cache misses and random access.
     *
     * The strategy has two phases:
     *  - Phase A (Outer Loops): Sorts the `free_indices` to optimize access
     *    to the *output* tensor.
     *  - Phase B (Inner Loops): Sorts the `contracted_indices` to optimize
     *    access to the *input* tensors during the reduction.
     *
     * Both phases use the same greedy concordance sorting logic.
     *
     * @return A single vector representing the full loop order.
     */
    static std::vector<Index> determine_loop_order(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const IndexClassification&                     idx_class)
    {
        std::vector<Index> order;
        order.reserve(idx_class.free_indices.size() +
                      idx_class.contracted_indices.size());

        // Phase A: Free indices (outer loops).
        auto free_ordered = greedy_concordance_sort(
            operands,
            std::unordered_set<Index>(idx_class.free_indices.begin(),
                                      idx_class.free_indices.end()));
        order.insert(order.end(), free_ordered.begin(), free_ordered.end());

        // Phase B: Contracted indices (inner loops).
        auto contr_ordered = greedy_concordance_sort(
            operands,
            std::unordered_set<Index>(idx_class.contracted_indices.begin(),
                                      idx_class.contracted_indices.end()));
        order.insert(order.end(), contr_ordered.begin(), contr_ordered.end());

        return order;
    }

    /**
     * @brief Performs a greedy sort on a set of candidate indices based on
     *        "concordance votes".
     *
     * In each step, the function calculates a score for each remaining candidate
     * index. The score is the sum of the NNZ of all operands that would be
     * accessed concordantly if that index were chosen next. An operand is
     * accessed concordantly if the chosen index is the first *un-ordered*
     * index in its physical storage layout.
     *
     * The index with the highest score is chosen, added to the order, and
     * removed from the candidate set. This process repeats until all candidates
     * are ordered.
     *
     * @param candidates The set of indices to be sorted.
     * @return A sorted vector of indices.
     */
    static std::vector<Index> greedy_concordance_sort(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        std::unordered_set<Index>                      candidates)
    {
        std::vector<Index> result;
        result.reserve(candidates.size());

        while (!candidates.empty()) {
            std::unordered_map<Index, double> votes;
            for (Index idx : candidates) votes[idx] = 0.0;

            // Each operand "votes" for the first candidate index it encounters
            // in its own physical mode order. The vote is weighted by the operand's NNZ.
            for (const auto& op : operands) {
                const std::vector<Index> mode_order = get_mode_order(op);
                for (Index idx : mode_order) {
                    if (candidates.count(idx)) {
                        votes[idx] += op->estimate_nnz();
                        break; // Only the first un-ordered index gets to vote.
                    }
                }
            }

            // Select the candidate with the highest total vote score.
            Index best = *candidates.begin();
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
    
    /**
     * @brief Chooses the optimal CSF storage format for the kernel's output tensor.
     *
     * The selection follows two main rules:
     *  1. Mode Order: The mode order of the output format is set to match the
     *     order of the `free_indices` in the kernel's loop order. This ensures
     *     that writes to the output tensor are fully concordant, which is
     *     critical for performance and avoids the need for a later transpose.
     *  2. Level Types: All levels are `COMPRESSED` by default. However, the
     *     innermost level is made `DENSE` if the output tensor's estimated
     *     density exceeds a certain threshold (e.g., 30%). This allows for
     *     efficient, vectorized writes in the tightest loop.
     *
     * @return A `StorageFormat` struct describing the chosen format.
     */
    static StorageFormat choose_output_format(
        const std::vector<Index>&             loop_order,
        const std::vector<Index>&             output_indices,
        double                                estimated_out_nnz,
        const std::unordered_map<Index, int>& dim_sizes)
    {
        // The mode order of the output format mirrors the free-index loop order
        // to guarantee concordant writes.
        const std::unordered_set<Index> out_set(
            output_indices.begin(), output_indices.end());

        std::vector<Index> mode_order;
        mode_order.reserve(output_indices.size());
        for (Index idx : loop_order) {
            if (out_set.count(idx)) mode_order.push_back(idx);
        }

        // Calculate the output tensor's density to inform the level type selection.
        double out_volume = 1.0;
        for (Index idx : mode_order) {
            auto it = dim_sizes.find(idx);
            out_volume *= (it != dim_sizes.end())
                ? static_cast<double>(it->second)
                : 100.0; // Fallback if dimension size is unknown.
        }
        const double density = (out_volume > 0.0)
            ? (estimated_out_nnz / out_volume)
            : 0.0;

        std::vector<LevelType> level_types;
        level_types.reserve(mode_order.size());
        for (size_t i = 0; i < mode_order.size(); ++i) {
            const bool is_innermost = (i == mode_order.size() - 1);
            // Use DENSE for the innermost level if density is high, enabling vectorization.
            // Otherwise, use COMPRESSED for memory efficiency.
            level_types.push_back(
                (is_innermost && density > 0.30)
                ? LevelType::DENSE
                : LevelType::COMPRESSED);
        }

        return StorageFormat{level_types, mode_order};
    }

    // =========================================================================
    // Kernel Cost Scoring
    // =========================================================================

    /**
     * @brief Calculates the total cost of a kernel based on a three-part model.
     *
     * The model includes:
     *  1. Compute Cost: A proxy for FLOPs, estimated as the sum of the NNZ of
     *     all input operands, scaled by a logarithmic factor to account for
     *     the overhead of N-way iterator coordination (branch divergence).
     *
     *  2. Access Penalty: A penalty applied for discordant memory access. This
     *     is calculated for each operand whose physical storage layout does
     *     not match the kernel's loop order.
     *
     *  3. Memory Write Cost: The cost of materializing the output tensor,
     *     proportional to its estimated NNZ. This often dominates on
     *     memory-bandwidth-bound accelerators.
     *
     * @return A `KernelCostBreakdown` struct with the detailed costs.
     */
    static KernelCostBreakdown score_kernel(
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>&                      loop_order,
        double                                         estimated_out_nnz)
    {
        // Tunable constants for the cost model, normalized to memory bandwidth units.
        constexpr double DISCORDANT_PENALTY_BASE  = 5.0;   // Exponential base for access penalty.
        constexpr double HBM_WRITE_FACTOR         = 1.5;   // Cost per output element written.
        constexpr double DIVERGENCE_LOG_FACTOR    = 0.15;  // Scales the N-way merge overhead.

        KernelCostBreakdown result;

        for (const auto& op : operands) {
            const double op_nnz      = op->estimate_nnz();
            result.compute_cost     += op_nnz;
            result.access_penalty   += compute_access_penalty(
                op, loop_order, op_nnz, DISCORDANT_PENALTY_BASE);
        }

        // The divergence factor models the increasing cost of coordinating more iterators.
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

    /**
     * @brief Computes the penalty for a single operand's discordant memory access.
     *
     * Access is "concordant" if the kernel's loop order iterates through the
     * operand's indices in the same order as its physical storage layout.
     *
     * If an index is visited out of order, a penalty is applied. The penalty
     * grows exponentially with the "inversion depth"—the distance in the loop
     * order between where the primary index should have been and where the
     * out-of-order secondary index was actually placed. This models the high
     * cost of breaking sequential access (e.g., forcing a binary search or
     * a full scan within a fiber).
     *
     * @return The calculated access penalty for the operand.
     */
    static double compute_access_penalty(
        const std::shared_ptr<ExprNode>& op,
        const std::vector<Index>&         loop_order,
        double                            op_nnz,
        double                            base)
    {
        const std::vector<Index> mode_order = get_mode_order(op);
        if (mode_order.size() < 2) return 0.0; // Single-mode tensors are always concordant.

        // Map each index to its position in the loop order.
        std::unordered_map<Index, int> loop_pos;
        for (int i = 0; i < static_cast<int>(loop_order.size()); ++i) {
            loop_pos[loop_order[i]] = i;
        }

        const Index primary = mode_order[0];
        auto prim_it = loop_pos.find(primary);
        if (prim_it == loop_pos.end()) return 0.0; // Primary index is not in this loop nest.
        const int primary_pos = prim_it->second;

        // Find the earliest secondary mode that appears *before* the primary mode.
        int earliest_secondary_pos = std::numeric_limits<int>::max();
        for (size_t m = 1; m < mode_order.size(); ++m) {
            auto sec_it = loop_pos.find(mode_order[m]);
            if (sec_it != loop_pos.end() && sec_it->second < primary_pos) {
                earliest_secondary_pos =
                    std::min(earliest_secondary_pos, sec_it->second);
            }
        }

        if (earliest_secondary_pos == std::numeric_limits<int>::max()) {
            return 0.0; // Fully concordant access.
        }

        // The penalty is exponential in the inversion depth.
        const int inversion_depth = primary_pos - earliest_secondary_pos;
        return op_nnz * std::pow(base, static_cast<double>(inversion_depth));
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * @brief A helper function to safely get the physical mode order from an
     *        expression node.
     *
     * This handles the polymorphism between a base `TensorNode` (which has a
     * physical format) and a `FusedContractionNode` (which has a designated
     * output format that acts as its physical layout for subsequent stages).
     *
     * @param op The expression node.
     * @return The physical mode order of the tensor or intermediate. Returns
     *         an empty vector if the node type is unrecognized.
     */
    static std::vector<Index> get_mode_order(
        const std::shared_ptr<ExprNode>& op)
    {
        if (auto t = std::dynamic_pointer_cast<TensorNode>(op)) {
            return t->format.mode_order;
        }
        if (auto f = std::dynamic_pointer_cast<FusedContractionNode>(op)) {
            return f->output_format.mode_order;
        }
        if (auto u = std::dynamic_pointer_cast<UnaryOpNode>(op)) {
            std::vector<Index> inner_order = get_mode_order(u->operand);
            std::vector<Index> valid_order;
            std::unordered_set<Index> out_indices(u->get_indices().begin(), u->get_indices().end());
            for (Index idx : inner_order) {
                if (out_indices.count(idx)) {
                    valid_order.push_back(idx);
                }
            }
            return valid_order;
        }
        return {};
    }

    static std::shared_ptr<TensorNode> get_underlying_tensor(const std::shared_ptr<ExprNode>& node) {
        if (auto t = std::dynamic_pointer_cast<TensorNode>(node)) return t;
        if (auto u = std::dynamic_pointer_cast<UnaryOpNode>(node)) return get_underlying_tensor(u->operand);
        return nullptr;
    }
};