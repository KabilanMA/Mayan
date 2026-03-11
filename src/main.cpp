// main.cpp
//
// Exercises the sparse tensor optimizer across four kernels of increasing
// complexity. For each kernel we:
//
//   1. Declare input tensors in COO format (mode_order = declaration order)
//   2. Show the "before" state: each tensor's original layout
//   3. Run DPOptimizer::optimize()
//   4. Print the full optimized plan via to_string()
//   5. Print per-tensor format recommendations and rationales
//
// Kernels tested:
//   A) SpGEMM        — A(i,j)   = B(i,k) * C(k,j)          [expect B=CSR, C=CSC]
//   B) Chain         — D(i,l)   = A(i,j) * B(j,k) * C(k,l) [tests pivot decomp]
//   C) SpTTM         — Y(i,j,r) = X(i,j,k) * M(k,r)        [3D tensor + matrix]
//   D) SpGEMM + HLL  — same as A but with real HLL sketches for NNZ estimation
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

// Quick stand-alone HLL estimate check (diagnostic only)
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

// ─────────────────────────────────────────────────────────────────────────────
// Test A: SpGEMM — A(i,j) = B(i,k) * C(k,j)
//
// The textbook motivating example.  B is accessed row-first (i outer) → CSR.
// C is accessed column-first (j outer) → must be transposed to CSC.
// The optimizer should discover loop order [i → j → k] and label B=CSR, C=CSC.
// ─────────────────────────────────────────────────────────────────────────────
static void test_spgemm()
{
    banner("Test A: SpGEMM   A(i,j) = B(i,k) * C(k,j)");

    // Dimensions
    const int M = 1000, K = 500, N = 800;

    // Tensors loaded as COO
    auto B = make_coo_tensor("B", {'i','k'}, {M, K}, 5000,  5000.0/(M*K));
    auto C = make_coo_tensor("C", {'k','j'}, {K, N}, 4000,  4000.0/(K*N));

    section("Before optimization");
    print_tensor_before(*B);
    print_tensor_before(*C);

    // Global output: A(i,j)
    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index, int> dims = {{'i',M},{'j',N},{'k',K}};

    auto plan = DPOptimizer::optimize({B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test B: 3-tensor chain — D(i,l) = A(i,j) * B(j,k) * C(k,l)
//
// A classic "matrix chain" in sparse form.  The optimal strategy is usually to
// contract one pair first (e.g. A*B → T(i,k)) then contract T*C.  Strategy C
// (pivot decomposition) should detect that j connects only A and B, and k
// connects only B and C, allowing the chain to be split.
//
// Each tensor is intentionally given different NNZ values so the cost model
// has a clear signal about which pair to contract first.
// ─────────────────────────────────────────────────────────────────────────────
static void test_chain()
{
    banner("Test B: Chain   D(i,l) = A(i,j) * B(j,k) * C(k,l)");

    const int I=400, J=800, K=200, L=600;

    // A is large, C is small — cost model should prefer contracting B*C first
    // to produce a small intermediate, then contract A with it.
    auto A = make_coo_tensor("A", {'i','j'}, {I,J}, 20000, 20000.0/(I*J));
    auto B = make_coo_tensor("B", {'j','k'}, {J,K}, 8000,   8000.0/(J*K));
    auto C = make_coo_tensor("C", {'k','l'}, {K,L}, 1200,   1200.0/(K*L));

    section("Before optimization");
    print_tensor_before(*A);
    print_tensor_before(*B);
    print_tensor_before(*C);

    const std::vector<Index> out  = {'i','l'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'k',K},{'l',L}};

    auto plan = DPOptimizer::optimize({A, B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test C: SpTTM — Y(i,j,r) = X(i,j,k) * M(k,r)
//
// Sparse Tensor-Times-Matrix: a 3-D sparse tensor X contracted with a dense(ish)
// matrix M along mode k.  Common in Tucker decomposition.
//
// X carries per-mode NNZ statistics in nnz_along_dim so that FormatSelector
// can make a DENSE vs COMPRESSED level decision for each CSF level.
// The k-mode of X is the contracted one; the optimizer should put k innermost.
// ─────────────────────────────────────────────────────────────────────────────
static void test_spttm()
{
    banner("Test C: SpTTM   Y(i,j,r) = X(i,j,k) * M(k,r)");

    const int I=50, J=60, K=40, R=10;

    // Build per-mode NNZ-per-slice stats for X
    SparseMetadata x_meta;
    x_meta.global_density = 0.05;
    // Simulate: each i-slice of X has ~120 nnz, each j-slice ~100, each k-slice ~150
    x_meta.nnz_along_dim['i'] = std::vector<double>(I, 120.0);
    x_meta.nnz_along_dim['j'] = std::vector<double>(J, 100.0);
    x_meta.nnz_along_dim['k'] = std::vector<double>(K, 150.0);

    // M is fairly dense (35%), enough to trigger DENSE innermost level
    SparseMetadata m_meta;
    m_meta.global_density = 0.35;
    m_meta.nnz_along_dim['k'] = std::vector<double>(K, static_cast<double>(R) * 0.35);
    m_meta.nnz_along_dim['r'] = std::vector<double>(R, static_cast<double>(K) * 0.35);

    auto X = std::make_shared<TensorNode>(
        "X", std::vector<Index>{'i','j','k'}, Shape{I,J,K},
        /*nnz=*/6000.0, StorageFormat{}, x_meta);

    auto M = std::make_shared<TensorNode>(
        "M", std::vector<Index>{'k','r'}, Shape{K,R},
        /*nnz=*/static_cast<double>(K*R)*0.35, StorageFormat{}, m_meta);

    section("Before optimization");
    print_tensor_before(*X);
    print_tensor_before(*M);

    const std::vector<Index> out = {'i','j','r'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'k',K},{'r',R}};

    auto plan = DPOptimizer::optimize({X, M}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test D: SpGEMM with real HLL sketches
//
// Repeats Test A but populates HLL sketches from simulated COO coordinate data.
// Demonstrates that:
//   (a) HLL::estimate() is within ~2% of the true NNZ
//   (b) The cost model uses the intersection estimate for output NNZ rather
//       than the density-product fallback, yielding a tighter bound
//   (c) The format decisions are identical to Test A (the HLL path doesn't
//       change which format wins — it only sharpens the NNZ estimate)
// ─────────────────────────────────────────────────────────────────────────────
static void test_spgemm_hll()
{
    banner("Test D: SpGEMM + HLL sketches   A(i,j) = B(i,k) * C(k,j)");

    const int M=1000, K=500, N=800;
    const long long B_nnz = 5000, C_nnz = 4000;

    // Build real HLL sketches from simulated coordinate streams
    section("Building HLL sketches from simulated COO data");
    auto B_sketch = build_hll_sketch(M, K, B_nnz, /*seed=*/42);
    auto C_sketch = build_hll_sketch(K, N, C_nnz, /*seed=*/99);

    verify_hll(B_sketch, B_nnz, "B");
    verify_hll(C_sketch, C_nnz, "C");

    // Demonstrate intersection estimate for output A
    {
        HyperLogLog hll_b(B_sketch), hll_c(C_sketch);
        const double intersection = HyperLogLog::estimate_intersection(hll_b, hll_c);
        std::cout << "  HLL intersection estimate (output NNZ upper bound): "
                  << static_cast<long long>(intersection) << "\n";
    }

    auto B = make_coo_tensor("B", {'i','k'}, {M,K}, B_nnz, (double)B_nnz/(M*K), B_sketch);
    auto C = make_coo_tensor("C", {'k','j'}, {K,N}, C_nnz, (double)C_nnz/(K*N), C_sketch);

    section("Before optimization");
    print_tensor_before(*B);
    print_tensor_before(*C);

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',M},{'j',N},{'k',K}};

    auto plan = DPOptimizer::optimize({B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test E: MTTKRP — A(i,r) = X(i,j,k) * B(j,r) * C(k,r)
//
// Matricised Tensor Times Khatri-Rao Product: the inner loop of the HOSVD /
// ALS algorithm for Tucker/CP decomposition.
//
// j and k are both contracted; r is the output rank dimension shared by B and C.
// The optimizer must decide whether to fuse all three or materialise an
// intermediate B⊙C Khatri-Rao product.  With small rank R the fused kernel
// typically wins; this test confirms the pivot heuristic doesn't force a
// premature split when N-ary fusion is cheaper.
// ─────────────────────────────────────────────────────────────────────────────
static void test_mttkrp()
{
    banner("Test E: MTTKRP   A(i,r) = X(i,j,k) * B(j,r) * C(k,r)");

    const int I=100, J=80, K=60, R=10;

    SparseMetadata x_meta;
    x_meta.global_density = 0.02;

    auto X = std::make_shared<TensorNode>(
        "X", std::vector<Index>{'i','j','k'}, Shape{I,J,K},
        /*nnz=*/static_cast<double>(I*J*K)*0.02, StorageFormat{}, x_meta);

    auto B = make_coo_tensor("B", {'j','r'}, {J,R}, (double)(J*R)*0.8, 0.8);
    auto C = make_coo_tensor("C", {'k','r'}, {K,R}, (double)(K*R)*0.8, 0.8);

    section("Before optimization");
    print_tensor_before(*X);
    print_tensor_before(*B);
    print_tensor_before(*C);

    const std::vector<Index> out = {'i','r'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J},{'k',K},{'r',R}};

    auto plan = DPOptimizer::optimize({X, B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("Format recommendations");
    print_format_recommendations(plan);
}

// Return the NnzEstimationMode of the root FusedContractionNode as a string
static std::string nnz_mode_str(const std::shared_ptr<ExprNode>& plan) {
    if (auto f = std::dynamic_pointer_cast<FusedContractionNode>(plan)) {
        // We need to re-run evaluate_fused to get the breakdown — instead,
        // we expose it by inspecting the plan's cached_nnz context.
        // (In production you'd store breakdown inside the node; here we just
        //  label it via a quick evaluate call for display purposes.)
        (void)f;
    }
    return "(see breakdown in evaluate_fused)";
}

// Helper: evaluate a plan's root kernel and print the NNZ mode
static void print_nnz_mode(
    const std::vector<std::shared_ptr<ExprNode>>& inputs,
    const std::vector<Index>& out,
    const std::unordered_map<Index,int>& dims)
{
    const IndexClassification cls = CostModel::classify_indices(inputs, out);
    NnzEstimationMode mode = NnzEstimationMode::DENSITY_PRODUCT_FALLBACK;
    // Re-evaluate just the NNZ path (free function exposed for testing)
    const FusedCostResult r = CostModel::evaluate_fused(inputs, out, dims);
    const char* labels[] = {
        "OUTER_PRODUCT",
        "HLL_SINGLE_COMPONENT",
        "HLL_MULTI_COMPONENT",
        "DENSITY_PRODUCT_FALLBACK"
    };
    const int idx = static_cast<int>(r.breakdown.nnz_mode);
    std::cout << "  NNZ estimation mode : " << labels[idx] << "\n"
              << "  Estimated output NNZ: " << static_cast<long long>(r.estimated_out_nnz) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test F: Outer product — A(i,j) = B(i) ⊗ C(j)   [no shared index at all]
//
// B and C have completely disjoint index sets; there is no contracted index.
// The correct output NNZ is NNZ(B) × NNZ(C).
// Old code would call estimate_intersection on different-space sketches →
// near-zero result.  New code detects contracted_indices.empty() and returns
// the Cartesian product directly.
// ─────────────────────────────────────────────────────────────────────────────
static void test_outer_product()
{
    banner("Test F: Outer product   A(i,j) = B(i) ⊗ C(j)   [no shared index]");

    const int I = 500, J = 400;
    const long long B_nnz = 300, C_nnz = 250;

    // Build HLL sketches
    auto B_sketch = build_hll_sketch(I, 1, B_nnz, 7);
    auto C_sketch = build_hll_sketch(J, 1, C_nnz, 13);

    section("HLL sketch accuracy");
    verify_hll(B_sketch, B_nnz, "B");
    verify_hll(C_sketch, C_nnz, "C");
    std::cout << "  Expected output NNZ (outer product): "
              << B_nnz * C_nnz << "\n";

    // Demonstrate old bug: raw intersection of different-space sketches
    {
        HyperLogLog hll_b(B_sketch), hll_c(C_sketch);
        const double wrong = HyperLogLog::estimate_intersection(hll_b, hll_c);
        std::cout << "  [BUG DEMO] Raw intersection (meaningless): "
                  << static_cast<long long>(wrong)
                  << "  ← near-zero by hash accident, NOT the correct answer\n";
    }

    auto B = make_coo_tensor("B", {'i'}, {I}, B_nnz, (double)B_nnz/I, B_sketch);
    auto C = make_coo_tensor("C", {'j'}, {J}, C_nnz, (double)C_nnz/J, C_sketch);

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J}};

    section("Before optimization");
    print_tensor_before(*B);
    print_tensor_before(*C);

    auto plan = DPOptimizer::optimize({B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("NNZ estimation");
    print_nnz_mode({B, C}, out, dims);

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test G: Multi-component — D(i,l) = A(i,j) * B(j,k) * C(l,m)
//
// A and B are connected via contracted index j.
// C has indices l (free) and m (contracted), but m appears only in C —
// C is an isolated component in the contraction graph.
//
// Correct formula:
//   NNZ(D) = NNZ(A*B) [HLL intersection] × NNZ(C) [outer product component]
//
// Old code would intersect ALL THREE sketches together → wrong result because
// C's (l,m) coordinate space has nothing to do with A's (i,j) or B's (j,k).
// ─────────────────────────────────────────────────────────────────────────────
static void test_multi_component()
{
    banner("Test G: Multi-component   D(i,l) = A(i,j) * B(j,k) * C(l,m)\n"
           "        [A-B connected via j; C is isolated]");

    const int I=200, J=150, K=100, L=300, M=80;
    const long long A_nnz=5000, B_nnz=3000, C_nnz=1000;

    auto A_sk = build_hll_sketch(I, J, A_nnz, 11);
    auto B_sk = build_hll_sketch(J, K, B_nnz, 22);
    auto C_sk = build_hll_sketch(L, M, C_nnz, 33);

    section("HLL sketch accuracy");
    verify_hll(A_sk, A_nnz, "A");
    verify_hll(B_sk, B_nnz, "B");
    verify_hll(C_sk, C_nnz, "C");

    // Show what old code would compute (wrong):
    {
        HyperLogLog ha(A_sk), hb(B_sk), hc(C_sk);
        std::vector<const HyperLogLog*> all3 = {&ha, &hb, &hc};
        const double wrong = HyperLogLog::estimate_intersection(all3);
        std::cout << "  [BUG DEMO] 3-way intersection of all sketches: "
                  << static_cast<long long>(wrong)
                  << "  ← ignores that C lives in a different space\n";
    }

    // Show what the corrected code computes:
    {
        HyperLogLog ha(A_sk), hb(B_sk), hc(C_sk);
        std::vector<const HyperLogLog*> ab = {&ha, &hb};
        const double ab_est  = HyperLogLog::estimate_intersection(ab);
        const double c_est   = hc.estimate();
        std::cout << "  [CORRECT] NNZ(A*B) ≈ " << static_cast<long long>(ab_est)
                  << "  NNZ(C) ≈ " << static_cast<long long>(c_est)
                  << "  Product ≈ " << static_cast<long long>(ab_est * c_est) << "\n";
    }

    auto A = make_coo_tensor("A", {'i','j'}, {I,J}, A_nnz, (double)A_nnz/(I*J), A_sk);
    auto B = make_coo_tensor("B", {'j','k'}, {J,K}, B_nnz, (double)B_nnz/(J*K), B_sk);
    auto C = make_coo_tensor("C", {'l','m'}, {L,M}, C_nnz, (double)C_nnz/(L*M), C_sk);

    // j and k are contracted (don't appear in output); m is also contracted;
    // i and l survive into output D(i,l)
    const std::vector<Index> out = {'i','l'};
    const std::unordered_map<Index,int> dims =
        {{'i',I},{'j',J},{'k',K},{'l',L},{'m',M}};

    section("Before optimization");
    print_tensor_before(*A);
    print_tensor_before(*B);
    print_tensor_before(*C);

    auto plan = DPOptimizer::optimize({A, B, C}, out, dims);

    section("Optimized plan");
    std::cout << plan->to_string() << "\n";

    section("NNZ estimation");
    print_nnz_mode({A, B, C}, out, dims);

    section("Format recommendations");
    print_format_recommendations(plan);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test H: Free-index overlap only — E(i,j) = A(i,j) + B(i,j)   [element-wise]
//
// Both tensors share i and j, but both are FREE (output) indices — there is no
// contracted index. This is a sparse addition / element-wise kernel.
//
// Correct output NNZ:
//   Union semantics → NNZ ≤ NNZ(A) + NNZ(B)   [use HLL union, not intersection]
//   Intersection semantics → NNZ ≤ min(NNZ(A), NNZ(B))
//
// Neither intersection NOR outer-product is correct for union semantics.
// This test confirms the outer-product path is triggered (contracted empty),
// and flags that for addition kernels the caller should use merge_union
// externally rather than evaluate_fused. We document this limitation clearly.
// ─────────────────────────────────────────────────────────────────────────────
static void test_free_index_overlap()
{
    banner("Test H: Element-wise addition   E(i,j) = A(i,j) + B(i,j)\n"
           "        [shared FREE indices only — no contraction]");

    const int I=300, J=400;
    const long long A_nnz=6000, B_nnz=5000;

    auto A_sk = build_hll_sketch(I, J, A_nnz, 55);
    auto B_sk = build_hll_sketch(I, J, B_nnz, 66);

    section("HLL sketch accuracy");
    verify_hll(A_sk, A_nnz, "A");
    verify_hll(B_sk, B_nnz, "B");

    // Correct union-semantics NNZ using HLL directly (outside cost model)
    {
        HyperLogLog ha(A_sk), hb(B_sk);
        const double union_est = ha.merge_union(hb).estimate();
        const double inter_est = HyperLogLog::estimate_intersection(ha, hb);
        std::cout << "  HLL union  estimate (add semantics) : "
                  << static_cast<long long>(union_est)  << "\n"
                  << "  HLL intersection (multiply/mask)    : "
                  << static_cast<long long>(inter_est) << "\n"
                  << "  NOTE: cost model triggers OUTER_PRODUCT path\n"
                  << "  (contracted_indices empty) → caller must choose\n"
                  << "  union or intersection externally for add kernels.\n";
    }

    auto A = make_coo_tensor("A", {'i','j'}, {I,J}, A_nnz, (double)A_nnz/(I*J), A_sk);
    auto B = make_coo_tensor("B", {'i','j'}, {I,J}, B_nnz, (double)B_nnz/(I*J), B_sk);

    const std::vector<Index> out = {'i','j'};
    const std::unordered_map<Index,int> dims = {{'i',I},{'j',J}};

    section("NNZ estimation (cost model — sees OUTER_PRODUCT path)");
    print_nnz_mode({A, B}, out, dims);
    std::cout << "  ↑ NNZ is over-estimated here because element-wise addition\n"
              << "    does not produce i*j output tuples. For SpAdd you should\n"
              << "    pre-compute NNZ via HLL union and supply it as cached_nnz.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    test_spgemm();
    test_chain();
    test_spttm();
    test_spgemm_hll();
    test_mttkrp();
    test_outer_product();
    test_multi_component();
    test_free_index_overlap();

    std::cout << "\n" << std::string(70,'=') << "\n"
              << "  All tests complete.\n"
              << std::string(70,'=') << "\n\n";
    return 0;
}