#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>

// Aliases for tensor dimensions and shapes
using Index = char;
using Shape = std::vector<int>;

// Holds statistical data for Sparse Cost Models
struct SparseMetadata {
    double global_density = 1.0; 
    
    // Degree-based bounding: NNZ per slice along a specific dimension.
    std::unordered_map<Index, std::vector<double>> nnz_along_dim;
    
    // HyperLogLog sketch for cardinality estimation
    std::vector<uint8_t> hll_sketch; 
};