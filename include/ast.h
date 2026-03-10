#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

using Index = char;
using Shape = std::vector<int>;

enum class LevelType { DENSE, COMPRESSED };

struct StorageFormat {
    std::vector<LevelType> level_types;
    std::vector<Index> mode_order;

    std::string to_string() const {
        std::string res = "[";
        for (size_t i = 0; i < mode_order.size(); ++i) {
            res += mode_order[i];
            res += (level_types[i] == LevelType::DENSE) ? "(D)" : "(C)";
            if (i < mode_order.size() - 1) res += ", ";
        }
        return res + "]";
    }
};

struct SparseMetadata {
    double global_density = 1.0;
    std::vector<u_int8_t> hll_sketch = {};
};

// Base AST Node
struct ExprNode {
    virtual ~ExprNode() = default;
    virtual std::vector<Index> get_indices() const = 0;
    virtual double estimate_nnz() const = 0;
    virtual std::string to_string() const = 0;
    SparseMetadata metadata;
};

// Leaf Node: Physical Tensor
struct TensorNode : public ExprNode {
    std::string name;
    std::vector<Index> indices;
    Shape shape;
    double nnz;
    StorageFormat format;

    TensorNode(std::string n, std::vector<Index> idx, Shape s, double non_zeros, 
               StorageFormat fmt, SparseMetadata meta = {})
        : name(std::move(n)), indices(std::move(idx)), shape(std::move(s)), 
          nnz(non_zeros), format(std::move(fmt)) {
        this->metadata = std::move(meta);
    }

    std::vector<Index> get_indices() const override { return indices; }
    double estimate_nnz() const override { return nnz; }
    std::string to_string() const override { 
        return name + format.to_string(); 
    }
};

// Internal Node: N-ary Fused Contraction
struct FusedContractionNode : public ExprNode {
    std::vector<std::shared_ptr<ExprNode>> operands;
    std::vector<Index> out_indices;
    double cached_nnz;
    std::vector<Index> loop_iteration_order; 
    StorageFormat output_format; 

    FusedContractionNode(std::vector<std::shared_ptr<ExprNode>> ops, 
                         std::vector<Index> out, double nnz,
                         std::vector<Index> loop_order, StorageFormat out_fmt)
        : operands(std::move(ops)), out_indices(std::move(out)), 
          cached_nnz(nnz), loop_iteration_order(std::move(loop_order)), 
          output_format(std::move(out_fmt)) {}

    std::vector<Index> get_indices() const override { return out_indices; }
    double estimate_nnz() const override { return cached_nnz; }
    
    std::string to_string() const override {
        std::string res = "Fused(\n  Operands: ";
        for (size_t i = 0; i < operands.size(); ++i) {
            res += operands[i]->to_string();
            if (i < operands.size() - 1) res += ", ";
        }
        res += "\n  Loops: ";
        for (char c : loop_iteration_order) res += c;
        res += "\n  Output Format: " + output_format.to_string() + "\n)";
        return res;
    }
};