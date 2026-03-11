#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>

using Index = char;
using Shape = std::vector<int>;

enum class LevelType { DENSE, COMPRESSED };

struct StorageFormat {
    std::vector<LevelType> level_types;
    std::vector<Index> mode_order;   // physical storage order of dimensions

    // e.g. "[i(C), k(C)]"
    std::string to_string() const {
        std::string res = "[";
        for (size_t i = 0; i < mode_order.size(); ++i) {
            res += mode_order[i];
            if (level_types[i] == LevelType::DENSE) {
                res += "(D)";
            } else if (level_types[i] == LevelType::COMPRESSED) {
                res += "(C)";
            } else {
                res += "(U_DEF)";
            }
            if (i < mode_order.size() - 1) res += ", ";
        }
        return res + "]";
    }

    bool operator==(const StorageFormat& o) const {
        return level_types == o.level_types && mode_order == o.mode_order;
    }

    bool operator!=(const StorageFormat& o) const { 
        return !(*this == o); 
    }
};

struct SparseMetadata {
    double global_density = 1.0;

    // Degree-based bounding: NNZ per slice along a specific dimension.
    // Populated by profiling or static analysis; used by FormatSelector to
    // decide DENSE vs COMPRESSED per level.
    std::unordered_map<Index, std::vector<double>> nnz_along_dim;

    // HyperLogLog sketch for cardinality estimation (see hll.h).
    // Empty -> cost model falls back to density-product heuristic.
    std::vector<uint8_t> hll_sketch;
};

// Base AST Node
struct ExprNode {
    virtual ~ExprNode() = default;
    virtual std::vector<Index> get_indices() const = 0;
    virtual double estimate_nnz()  const = 0;
    virtual std::string to_string()     const = 0;
    SparseMetadata metadata;
};

// ─── Leaf Node: Physical Tensor ───────────────────────────────────────────────
//
// `indices`      : the LOGICAL indices in the original einsum notation.
//                  e.g. B(i,k) → indices = {'i','k'}
//                  These never change; they describe the mathematical role.
//
// `format`       : the PHYSICAL storage layout.
//                  format.mode_order is the actual memory order of the tensor.
//                  Before optimization this matches `indices`.
//                  After optimization (apply_recommended_formats) it is updated
//                  to the concordant order selected by FormatSelector.
//
// `format_label` : human-readable format name set by the optimizer.
//                  e.g. "CSR", "CSC", "CSF[j,i,k]"
//                  Empty before optimization.
//
// `rationale`    : one-sentence explanation of why this format was chosen.
//                  Empty before optimization.
// ─────────────────────────────────────────────────────────────────────────────
struct TensorNode : public ExprNode {
    std::string        name;
    std::vector<Index> indices;      // logical, original einsum order (immutable)
    Shape              shape;
    double             nnz;
    StorageFormat      format;       // physical layout; mode_order updated by optimizer
    std::string        format_label; // "CSR" / "CSC" / "CSF[...]" — set post-optimization
    std::string        rationale;    // why this format — set post-optimization

    TensorNode(std::string n, std::vector<Index> idx, Shape s, double non_zeros,
               StorageFormat fmt, SparseMetadata meta = {})
        : name(std::move(n)), indices(std::move(idx)), shape(std::move(s)),
          nnz(non_zeros), format(std::move(fmt))
    {
        this->metadata = std::move(meta);
        // Default: format.mode_order should match `indices` for a fresh tensor.
        // If the caller didn't set it, initialise it now.
        if (this->format.mode_order.empty()) {
            this->format.mode_order = this->indices;
            this->format.level_types.assign(this->indices.size(), LevelType::COMPRESSED);
        }
    }

    std::vector<Index> get_indices() const override { return indices; }
    double             estimate_nnz() const override { return nnz; }

    // -------------------------------------------------------------------------
    // to_string(): displays indices in the PHYSICAL (optimized) mode order.
    //
    // Before optimization: mode_order == indices, so output looks the same
    //   as the einsum notation: "B(i,k):[i(C),k(C)]"
    //
    // After optimization: mode_order reflects the chosen CSF layout.
    //   SpGEMM example:
    //     B(i,k) stored as CSR → "B(i,k):[i(C),k(C)]  — CSR"   (unchanged)
    //     C(k,j) stored as CSC → "C(j,k):[j(C),k(C)]  — CSC"   (reordered!)
    //
    // The name always shows the optimized physical order first, followed by
    // the label and the storage-format annotation.
    // -------------------------------------------------------------------------
    std::string to_string() const override {
        // Physical (optimized) index order — what the hardware actually sees
        const std::vector<Index>& phys = format.mode_order.empty()
                                         ? indices
                                         : format.mode_order;

        std::string res = name + "(";
        for (size_t i = 0; i < phys.size(); ++i) {
            res += phys[i];
            if (i < phys.size() - 1) res += ",";
        }
        res += "):" + format.to_string();

        if (!format_label.empty()) {
            res += "  [" + format_label + "]";
        }
        return res;
    }
};

// ─── Internal Node: N-ary Fused Contraction ──────────────────────────────────
//
// Represents one kernel in the execution plan.
//
// `operands`             : the sub-expressions this kernel reads
// `out_indices`          : logical output indices (after contracting the rest)
// `cached_nnz`           : estimated NNZ of the output tensor
// `loop_iteration_order` : physical loop nest (free indices first, then contracted)
// `output_format`        : recommended CSF format for the materialized output
// ─────────────────────────────────────────────────────────────────────────────
struct FusedContractionNode : public ExprNode {
    std::vector<std::shared_ptr<ExprNode>> operands;
    std::vector<Index>                     out_indices;
    double                                 cached_nnz;
    std::vector<Index>                     loop_iteration_order;
    StorageFormat                          output_format;

    FusedContractionNode(std::vector<std::shared_ptr<ExprNode>> ops,
                         std::vector<Index> out, double nnz,
                         std::vector<Index> loop_order, StorageFormat out_fmt)
        : operands(std::move(ops)), out_indices(std::move(out)),
          cached_nnz(nnz), loop_iteration_order(std::move(loop_order)),
          output_format(std::move(out_fmt)) {}

    std::vector<Index> get_indices() const override { return out_indices; }
    double             estimate_nnz() const override { return cached_nnz; }

    // -------------------------------------------------------------------------
    // to_string(): shows the full kernel plan.
    //
    // Prints:
    //   1. Einsum expression with indices in their PHYSICAL (optimized) order
    //   2. Loop nest in execution order
    //   3. Output storage format
    //   4. Each operand's detail (recursive), indented
    //
    // SpGEMM example output:
    //
    //   Fused(
    //     Einsum: A(i,j) = B(i,k) · C(j,k)
    //     Loops: i → j → k
    //     Output: [i(C), j(C)]  ~5000 nnz
    //     Inputs:
    //       B(i,k):[i(C),k(C)]  [CSR]
    //       C(j,k):[j(C),k(C)]  [CSC]
    //   )
    // -------------------------------------------------------------------------
    std::string to_string() const override {
        std::string res = "Fused(\n";

        // ── 1. Einsum string ─────────────────────────────────────────────────
        // Output side: use output_format.mode_order (physical output order)
        res += "  Einsum: out(";
        const auto& out_phys = output_format.mode_order.empty()
                               ? out_indices
                               : output_format.mode_order;
        for (size_t i = 0; i < out_phys.size(); ++i) {
            res += out_phys[i];
            if (i < out_phys.size() - 1) res += ",";
        }
        res += ") =";

        // Input side: each operand in its physical index order
        for (size_t i = 0; i < operands.size(); ++i) {
            res += " ";
            // For leaf tensors, show the physical (optimized) index order
            if (auto t = std::dynamic_pointer_cast<TensorNode>(operands[i])) {
                const auto& phys = t->format.mode_order.empty()
                                   ? t->indices
                                   : t->format.mode_order;
                res += t->name + "(";
                for (size_t j = 0; j < phys.size(); ++j) {
                    res += phys[j];
                    if (j < phys.size() - 1) res += ",";
                }
                res += ")";
            } else {
                // Intermediate result: label it as "T" + operand index
                res += "T" + std::to_string(i) + "(";
                const auto idxs = operands[i]->get_indices();
                for (size_t j = 0; j < idxs.size(); ++j) {
                    res += idxs[j];
                    if (j < idxs.size() - 1) res += ",";
                }
                res += ")";
            }
            if (i < operands.size() - 1) res += " ·";
        }
        res += "\n";

        // ── 2. Loop nest ─────────────────────────────────────────────────────
        res += "  Loops: ";
        for (size_t i = 0; i < loop_iteration_order.size(); ++i) {
            res += loop_iteration_order[i];
            if (i < loop_iteration_order.size() - 1) res += " → ";
        }
        res += "\n";

        // ── 3. Output format + estimated NNZ ─────────────────────────────────
        res += "  Output: " + output_format.to_string()
             + "  (~" + std::to_string(static_cast<long long>(cached_nnz)) + " nnz)\n";

        // ── 4. Operand details (recursive, indented) ──────────────────────────
        res += "  Inputs:\n";
        for (const auto& op : operands) {
            // Indent the sub-tree by two extra spaces
            const std::string sub = op->to_string();
            res += "    ";
            for (char c : sub) {
                res += c;
                if (c == '\n') res += "    ";
            }
            res += "\n";
        }

        res += ")";
        return res;
    }
};