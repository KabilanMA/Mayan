# Mayan

### APIs

```cpp
static std::vector<Index> CostModel::infer_output_indices(
        const std::vector<std::shared_ptr<ExprNode>>& subset_operands,
        const std::vector<std::shared_ptr<ExprNode>>& all_operands,
        const std::vector<Index>& global_out_indices)
```

Given a subset of tensors, this will find indices that should be output of this subset contraction. For example let's say we have chained-SpGeMM, A(i,j) * B(k,j) * C(k,l) = D(i,l), and assume the subset operands are A(i,j) and B(k,j), then the output indices returned by this function will be {i,k}. Why? i is in the final output tensor D(i,l) and k is in the other input operands other than A(i,j) and B(k,j), that is C(k,l). j is not in the output indices because it is only present in tensor A and B, therefore it can (Not necessarily needs to be. Other optimizations can relax it if needed) be contracted away.

---

```cpp
static void evaluate_nary_fusion(
        DPState& state,
        const std::vector<std::shared_ptr<ExprNode>>& operands,
        const std::vector<Index>& out_indices,
        const std::unordered_map<Index, int>& dim_sizes)
```

