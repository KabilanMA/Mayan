// main.cpp
//
// Build (C++17):
//   g++ -std=c++17 -O2 -o optimizer main.cpp
//
// All headers are expected alongside this file:
//   ast.h  hll.h  format_selector.h  cost_model.h  dp_optimizer.h

#include "ast.h"
#include "hll.h"
#include "format_selector.h"
#include "cost_model.h"
#include "dp_optimizer.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <cstdint>

// ─── Pretty-print helpers ─────────────────────────────────────────────────────

static void banner(const std::string& title) {
    const std::string bar(70, '=');
    std::cout << "\n" << bar << "\n"
              << "  " << title << "\n"
              << bar << "\n";
}

static void section(const std::string& title) {
    std::cout << "\n── " << title << " ──\n";
}

// Print one tensor's "before optimization" state
static void print_tensor_before(const TensorNode& t) {
    std::cout << "  " << t.name << "(";
    for (size_t i = 0; i < t.indices.size(); ++i) {
        std::cout << t.indices[i];
        if (i < t.indices.size() - 1) std::cout << ",";
    }
    std::cout << ")  NNZ=" << static_cast<long long>(t.nnz)
              << "  density=" << std::fixed << std::setprecision(4)
              << t.metadata.global_density
              << "  layout=" << t.format.to_string()
              << "  hll=" << (t.metadata.hll_sketch.empty() ? "none" : "populated")
              << "\n";
}

// Walk the final AST and print each leaf tensor's format recommendation
static void print_format_recommendations(const std::shared_ptr<ExprNode>& node,
                                          int depth = 0)
{
    if (auto t = std::dynamic_pointer_cast<TensorNode>(node)) {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "  Tensor " << t->name
                  << " → format: " << (t->format_label.empty() ? "?" : t->format_label)
                  << "\n";
        if (!t->rationale.empty()) {
            std::cout << indent << "    " << t->rationale << "\n";
        }
        return;
    }
    if (auto u = std::dynamic_pointer_cast<UnaryOpNode>(node)) {
        print_format_recommendations(u->operand, depth);
        return;
    }
    if (auto f = std::dynamic_pointer_cast<FusedContractionNode>(node)) {
        for (const auto& op : f->operands) {
            print_format_recommendations(op, depth + 1);
        }
    }
}

// ─── HLL sketch builder ───────────────────────────────────────────────────────
//
// Simulate a sparse tensor's coordinate set by hashing random (row, col) pairs
// into a HyperLogLog sketch.  In a real system this runs during the COO load.
//
//   dim0, dim1  : dimension sizes for the two modes
//   nnz         : number of unique non-zero coordinates to simulate
//   seed        : RNG seed for reproducibility
//
static std::vector<uint8_t> build_hll_sketch(int dim0, int dim1,
                                              long long nnz, uint64_t seed)
{
    HyperLogLog hll;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, dim0 - 1);
    std::uniform_int_distribution<int> col_dist(0, dim1 - 1);

    // Insert `nnz` distinct-ish (i, j) coordinate pairs
    for (long long n = 0; n < nnz; ++n) {
        const uint64_t r = static_cast<uint64_t>(row_dist(rng));
        const uint64_t c = static_cast<uint64_t>(col_dist(rng));
        // Combine the two coordinate components into one hash
        hll.add(hll_hash(hll_hash_combine(hll_hash(r), hll_hash(c))));
    }

    return hll.serialize();
}

// Build a per-mode HLL sketch: inserts only the coordinate along `mode_dim`
// (ignoring all other dimensions). The result estimates distinct values along
// that one axis — exactly what projected_nnz needs for self-reduction.
static std::vector<uint8_t> build_mode_sketch(int mode_dim, long long nnz, uint64_t seed)
{
    HyperLogLog hll;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> coord_dist(0, mode_dim - 1);
    for (long long n = 0; n < nnz; ++n) {
        hll.add(hll_hash(static_cast<uint64_t>(coord_dist(rng))));
    }
    return hll.serialize();
}

// Build a KMV mode sketch for N-way intersection robustness
static std::vector<uint64_t> build_mode_kmv_sketch(int mode_dim, long long nnz, uint64_t seed) {
    KMinValues kmv;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> coord_dist(0, mode_dim - 1);
    for (long long n = 0; n < nnz; ++n) {
        kmv.add(hll_hash(static_cast<uint64_t>(coord_dist(rng))));
    }
    return kmv.serialize();
}

// Build a full-tuple KMV sketch for element-wise operations
static std::vector<uint64_t> build_kmv_sketch(int dim0, int dim1,
                                              long long nnz, uint64_t seed)
{
    KMinValues kmv;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, dim0 - 1);
    std::uniform_int_distribution<int> col_dist(0, dim1 - 1);

    for (long long n = 0; n < nnz; ++n) {
        const uint64_t r = static_cast<uint64_t>(row_dist(rng));
        const uint64_t c = static_cast<uint64_t>(col_dist(rng));
        kmv.add(hll_hash(hll_hash_combine(hll_hash(r), hll_hash(c))));
    }

    return kmv.serialize();
}

static void verify_hll(const std::vector<uint8_t>& sketch, long long true_nnz,
                        const std::string& name)
{
    HyperLogLog hll(sketch);
    const double est = hll.estimate();
    const double err = std::abs(est - static_cast<double>(true_nnz))
                     / static_cast<double>(true_nnz) * 100.0;
    std::cout << "  HLL estimate for " << name
              << ": " << static_cast<long long>(est)
              << "  (true=" << true_nnz
              << ", error=" << std::fixed << std::setprecision(1) << err << "%)\n";
}

// ─── Helpers to build "COO" TensorNodes ──────────────────────────────────────
//
// A fresh COO tensor has an empty StorageFormat; the TensorNode constructor
// fills mode_order = indices and all levels = COMPRESSED.  This models a
// tensor that was just loaded from a .mtx / .tns file in coordinate order.

static std::shared_ptr<TensorNode> make_coo_tensor(
    const std::string&        name,
    std::vector<Index>        indices,
    Shape                     shape,
    double                    nnz,
    double                    density,
    std::vector<uint8_t>      hll_sketch = {})
{
    SparseMetadata meta;
    meta.global_density = density;
    meta.hll_sketch     = std::move(hll_sketch);

    // Empty StorageFormat → constructor sets mode_order = indices (COO default)
    return std::make_shared<TensorNode>(
        name, std::move(indices), std::move(shape),
        nnz, StorageFormat{}, std::move(meta));
}

// Helper: evaluate a plan's root kernel and print the NNZ mode
static void print_nnz_mode(
    const std::vector<std::shared_ptr<ExprNode>>& inputs,
    const std::vector<Index>& out,
    const std::unordered_map<Index,int>& dims)
{
    const FusedCostResult r = CostModel::evaluate_fused(inputs, out, dims);
    const char* labels[] = {
        "OUTER_PRODUCT",
        "BROADCAST_JOIN",
        "KMV_INTERSECTION_SINGLE_COMPONENT",
        "KMV_INTERSECTION_MULTI_COMPONENT",
        "DENSITY_PRODUCT_FALLBACK"
    };
    const int idx = static_cast<int>(r.breakdown.nnz_mode);
    std::cout << "  NNZ estimation mode : " << labels[idx] << "\n"
              << "  Estimated output NNZ: " << static_cast<long long>(r.estimated_out_nnz) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test I: Element-wise multiplication — E(i,j) = A(i,j) * B(i,j)
//
// Both tensors share i and j, and both are FREE indices (no contraction).
// The optimizer identifies this as an element-wise operation (is_element_wise)
// and uses the KMV full-tuple sketch intersection to accurately bound
// the Hadamard product overlap, rather than falling back to an outer product.
// ─────────────────────────────────────────────────────────────────────────────
static void test_elementwise_multiplication()
{
    banner("Test I: Element-wise multiplication   E(i,j) = A(i,j) * B(i,j)\n"
           "        [Uses KMV full-tuple intersection]");

    const int I=300, J=400;
    const long long A_nnz=6000, B_nnz=5000;

    auto A_kmv = build_kmv_sketch(I, J, A_nnz, 42);
    auto B_kmv = build_kmv_sketch(I, J, B_nnz, 99); 

    SparseMetadata A_meta, B_meta;
    A_meta.global_density = (double)A_nnz/(I*J);
    A_meta.kmv_sketch = A_kmv;
    
    B_meta.global_density = (double)B_nnz/(I*J);
    B_meta.kmv_sketch = B_kmv;

    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i','j'}, Shape{I,J}, A_nnz, StorageFormat{}, A_meta);
    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'i','j'}, Shape{I,J}, B_nnz, StorageFormat{}, B_meta);

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J}};

    std::vector<std::shared_ptr<ExprNode>> inputs = {A, B};

    // CanonicalizationPass pass(out, dims);
    // for (auto& input : inputs) {
    //     input = pass.mutate(input); 
    // }

    section("NNZ estimation (cost model — sees element-wise path)");
    print_nnz_mode(inputs, out, dims);

    auto plan = DPOptimizer::optimize(inputs, out, dims);
    section("Optimized plan");
    std::cout << plan->to_string() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test J: 4-Way Multi-Join — R(i,j,l,m) = A(i,k) * B(j,k) * C(l,k) * D(m,k)
//
// All 4 tensors share the contracted index `k` ("Star schema" join). 
// This tests the robustness of KMinValues (Theta sketches) over HLL for 
// estimating deep multi-way joins natively without mathematical blowup.
// ─────────────────────────────────────────────────────────────────────────────
static void test_multiway_join()
{
    banner("Test J: 4-Way Multi-Join   R(i,j,l,m) = A(i,k) * B(j,k) * C(l,k) * D(m,k)\n"
           "        [Uses KMV mode-sketch N-way intersection]");

    const int I=100, J=100, L=100, M=100, K=50000;
    const long long nnz = 10000;

    // Give them all mode kmv sketches for 'k' using different seeds
    auto skA = build_mode_kmv_sketch(K, nnz, 1);
    auto skB = build_mode_kmv_sketch(K, nnz, 2);
    auto skC = build_mode_kmv_sketch(K, nnz, 3);
    auto skD = build_mode_kmv_sketch(K, nnz, 4);

    SparseMetadata metaA, metaB, metaC, metaD;
    metaA.mode_kmv_sketches = {{'k', skA}};
    metaB.mode_kmv_sketches = {{'k', skB}};
    metaC.mode_kmv_sketches = {{'k', skC}};
    metaD.mode_kmv_sketches = {{'k', skD}};

    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i','k'}, Shape{I,K}, nnz, StorageFormat{}, metaA);
    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'j','k'}, Shape{J,K}, nnz, StorageFormat{}, metaB);
    auto C = std::make_shared<TensorNode>("C", std::vector<Index>{'l','k'}, Shape{L,K}, nnz, StorageFormat{}, metaC);
    auto D = std::make_shared<TensorNode>("D", std::vector<Index>{'m','k'}, Shape{M,K}, nnz, StorageFormat{}, metaD);

    const std::vector<Index> out = {'i','j','l','m'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'l',L},{'m',M},{'k',K}};

    section("NNZ estimation (cost model — KMV Multi-Way Join path)");
    print_nnz_mode({A, B, C, D}, out, dims);
    
    auto plan = DPOptimizer::optimize({A, B, C, D}, out, dims);
    section("Optimized plan");
    std::cout << plan->to_string() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test K: Pure Outer Product — C(i,j) = A(i) * B(j)
//
// Tensors have no shared indices and no indices are contracted.
// This should be identified as a pure outer product.
// ─────────────────────────────────────────────────────────────────────────────
static void test_pure_outer_product()
{
    banner("Test K: Pure Outer Product   C(i,j) = A(i) * B(j)\n"
           "        [Uses OUTER_PRODUCT estimation]");

    const int I=100, J=200;
    const long long A_nnz=50, B_nnz=60;

    SparseMetadata A_meta, B_meta;
    A_meta.global_density = (double)A_nnz/I;
    B_meta.global_density = (double)B_nnz/J;

    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i'}, Shape{I}, A_nnz, StorageFormat{}, A_meta);
    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'j'}, Shape{J}, B_nnz, StorageFormat{}, B_meta);

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J}};

    section("NNZ estimation");
    print_nnz_mode({A, B}, out, dims);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test L: Self-Contraction (Projection) — B(i) = sum_j A(i,j)
//
// One index ('j') of a tensor is summed out. This tests the `projected_nnz`
// logic, which should use per-mode HLL sketches for accuracy.
// ─────────────────────────────────────────────────────────────────────────────
static void test_self_contraction()
{
    banner("Test L: Self-Contraction (Projection)   B(i) = sum_j A(i,j)\n"
           "        [Uses per-mode HLL for projected NNZ]");
    
    const int I=1000, J=500;
    const long long A_nnz=20000;
    const long long true_i_cardinality = 950; // A has 950 unique 'i' values

    SparseMetadata A_meta;
    A_meta.global_density = (double)A_nnz/(I*J);
    // Provide a per-mode sketch for the surviving dimension 'i'
    A_meta.mode_sketches['i'] = build_mode_sketch(I, true_i_cardinality, 123);

    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i','j'}, Shape{I,J}, A_nnz, StorageFormat{}, A_meta);

    const std::vector<Index> out = {'i'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J}};

    section("NNZ estimation");
    print_nnz_mode({A}, out, dims);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test M: Broadcast / Partial Join — C(i,j,k) = A(i,j) * B(j,k)
//
// Indices are shared ('j') but not contracted. Tests the BROADCAST_JOIN path.
// ─────────────────────────────────────────────────────────────────────────────
static void test_broadcast_join()
{
    banner("Test M: Broadcast / Partial Join   C(i,j,k) = A(i,j) * B(j,k)\n"
           "        [Uses BROADCAST_JOIN estimation]");

    const int I=100, J=200, K=300;
    const long long A_nnz=1000, B_nnz=2000;
    const long long A_j_card = 180; // A has 180 unique 'j' values
    const long long B_j_card = 190; // B has 190 unique 'j' values

    // --- With sketches ---
    SparseMetadata A_meta, B_meta;
    A_meta.global_density = (double)A_nnz/(I*J);
    A_meta.mode_kmv_sketches['j'] = build_mode_kmv_sketch(J, A_j_card, 42);

    B_meta.global_density = (double)B_nnz/(J*K);
    B_meta.mode_kmv_sketches['j'] = build_mode_kmv_sketch(J, B_j_card, 99);

    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i','j'}, Shape{I,J}, A_nnz, StorageFormat{}, A_meta);
    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'j','k'}, Shape{J,K}, B_nnz, StorageFormat{}, B_meta);

    const std::vector<Index> out = {'i','j','k'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'k',K}};

    section("NNZ estimation (with KMV sketches)");
    print_nnz_mode({A, B}, out, dims);
    
    // --- Without sketches (fallback) ---
    auto A_no_sketch = std::make_shared<TensorNode>("A", std::vector<Index>{'i','j'}, Shape{I,J}, A_nnz, StorageFormat{}, SparseMetadata());
    auto B_no_sketch = std::make_shared<TensorNode>("B", std::vector<Index>{'j','k'}, Shape{J,K}, B_nnz, StorageFormat{}, SparseMetadata());
    
    section("NNZ estimation (no sketches - fallback path)");
    print_nnz_mode({A_no_sketch, B_no_sketch}, out, dims);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test N: Diagonal Trace — A(i,j) = B(i,k,k) * C(j)
//
// Tests the AST canonicalization pass converting repeated indices into an
// explicit TraceNode, shielding the CostModel from handling trace semantics.
// ─────────────────────────────────────────────────────────────────────────────
static void test_diagonal_trace()
{
    banner("Test N: Diagonal Trace   A(i,j) = B(i,k,k) * C(j)\n"
           "        [Uses AST Canonicalization Pass]");

    const int I=100, J=200, K=50;
    const long long B_nnz = 5000, C_nnz = 150;

    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'i','k','k'}, Shape{I,K,K}, B_nnz, StorageFormat{});
    auto C = std::make_shared<TensorNode>("C", std::vector<Index>{'j'}, Shape{J}, C_nnz, StorageFormat{});

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'k',K}};

    std::vector<std::shared_ptr<ExprNode>> inputs = {B, C};

    CanonicalizationPass pass(out, dims);
    for (auto& input : inputs) {
        input = pass.mutate(input); 
    }

    section("AST after Canonicalization (Passed to DP Optimizer)");
    std::cout << "  " << inputs[0]->to_string() << "\n";
    std::cout << "  " << inputs[1]->to_string() << "\n";

    auto plan = DPOptimizer::optimize(inputs, out, dims);
    section("Optimized plan");
    std::cout << plan->to_string() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    test_elementwise_multiplication();
    // test_multiway_join();
    // test_pure_outer_product();
    // test_self_contraction();
    // test_broadcast_join();
    // test_diagonal_trace();

    std::cout << "\n" << std::string(70,'=') << "\n"
              << "  All tests complete.\n"
              << std::string(70,'=') << "\n\n";
    return 0;
}