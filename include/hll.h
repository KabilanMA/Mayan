#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <cassert>

// =============================================================================
// HyperLogLog Cardinality Estimator
//
// Used by the cost model to estimate output NNZ for sparse tensor contractions
// without materializing intermediate results. The key operations needed are:
//
//   - merge_union  : |A ∪ B| — used for output NNZ of addition-like ops
//   - estimate_intersection : |A ∩ B| — used for contraction (join) output NNZ
//   - estimate_projection   : |π_k(T)| — used to estimate the cardinality of
//                             a single index dimension after contraction
//
// Precision is controlled by PRECISION (p). Using p=14 gives 16384 registers
// with ~0.8% standard error, which is more than sufficient for cost model
// heuristics where we only need order-of-magnitude accuracy.
// =============================================================================

class HyperLogLog {
public:
    // p=14 → m=16384 registers, ~0.8% standard error
    static constexpr int PRECISION     = 14;
    static constexpr int NUM_REGISTERS = 1 << PRECISION;
    // Alpha_m bias correction constant (asymptotic formula valid for m >= 128)
    static constexpr double ALPHA_M = 0.7213 / (1.0 + 1.079 / NUM_REGISTERS);

    // ----- Construction ------------------------------------------------------

    HyperLogLog() : regs_(NUM_REGISTERS, 0) {}

    // Restore from a serialized sketch (e.g., stored in SparseMetadata::hll_sketch)
    explicit HyperLogLog(const std::vector<uint8_t>& sketch) {
        if (static_cast<int>(sketch.size()) == NUM_REGISTERS) {
            regs_ = sketch;
        } else {
            // Size mismatch: fall back to empty sketch rather than UB
            regs_.assign(NUM_REGISTERS, 0);
        }
    }

    // ----- Insertion ---------------------------------------------------------
    // Add a pre-hashed 64-bit value. Callers should hash their domain keys
    // with hll_hash() before calling this.
    void add(uint64_t hashed_value) {
        // Top PRECISION bits select the register
        const uint32_t reg_idx = static_cast<uint32_t>(hashed_value >> (64 - PRECISION));
        // Remaining bits: position of the leftmost 1-bit (ρ function)
        const uint64_t w = hashed_value << PRECISION;
        const uint8_t  rho = (w == 0)
            ? static_cast<uint8_t>(65 - PRECISION)   // all zeros → max ρ
            : static_cast<uint8_t>(__builtin_clzll(w) + 1);
        regs_[reg_idx] = std::max(regs_[reg_idx], rho);
    }

    // ----- Cardinality Estimation --------------------------------------------

    double estimate() const {
        // Raw HyperLogLog estimator
        double sum = 0.0;
        int zeros = 0;
        for (uint8_t r : regs_) {
            sum += std::ldexp(1.0, -static_cast<int>(r)); // 2^(-r)
            if (r == 0) ++zeros;
        }
        double E = ALPHA_M * static_cast<double>(NUM_REGISTERS) *
                   static_cast<double>(NUM_REGISTERS) / sum;

        // Small-range correction: linear counting when E is small and there
        // are empty registers (regime where HLL underestimates)
        if (E <= 2.5 * NUM_REGISTERS && zeros > 0) {
            E = static_cast<double>(NUM_REGISTERS) *
                std::log(static_cast<double>(NUM_REGISTERS) /
                         static_cast<double>(zeros));
        }

        // Large-range correction: compensate for 32-bit hash space collisions
        // Only relevant if using 32-bit hashes; with 64-bit hashes this rarely fires
        constexpr double TWO_32 = 4294967296.0;
        if (E > TWO_32 / 30.0) {
            E = -TWO_32 * std::log(1.0 - E / TWO_32);
        }

        return std::max(E, 1.0); // cardinality is at least 1 for a non-empty set
    }

    // ----- Set Operations ----------------------------------------------------

    // Union: max-register merge. Resulting sketch estimates |A ∪ B|.
    // Used for addition-semantics output NNZ and for building union bounds.
    [[nodiscard]] HyperLogLog merge_union(const HyperLogLog& other) const {
        HyperLogLog result;
        for (int i = 0; i < NUM_REGISTERS; ++i) {
            result.regs_[i] = std::max(regs_[i], other.regs_[i]);
        }
        return result;
    }

    // Intersection estimate via inclusion-exclusion:
    //   |A ∩ B| ≈ |A| + |B| - |A ∪ B|
    //
    // This is an approximation (not exact), but it's the standard HLL approach
    // and is sufficient for cost model purposes. For n-way intersections use the
    // overload below.
    [[nodiscard]] static double estimate_intersection(
        const HyperLogLog& a, const HyperLogLog& b)
    {
        const double card_a     = a.estimate();
        const double card_b     = b.estimate();
        const double card_union = a.merge_union(b).estimate();
        // Inclusion-exclusion; clamp to [1, min(|A|, |B|)]
        const double raw = card_a + card_b - card_union;
        return std::max(1.0, std::min(raw, std::min(card_a, card_b)));
    }

    // N-way intersection estimate via successive pairwise inclusion-exclusion.
    // For sparse tensors: the result of contracting N tensors is bounded by the
    // intersection of the coordinate sets of their shared indices.
    [[nodiscard]] static double estimate_intersection(
        const std::vector<const HyperLogLog*>& sketches)
    {
        if (sketches.empty()) 
            return 0.0;
        if (sketches.size() == 1) 
            return sketches[0]->estimate();

        // The iterative inclusion-exclusion formula breaks down mathematically
        // for N >= 3 because it improperly subtracts sets.
        // Instead, we compute an upper bound using the minimum of pairwise 
        // intersections, which safely and accurately bounds multi-way overlap.
        double result = estimate_intersection(*sketches[0], *sketches[1]);
        for (size_t i = 2; i < sketches.size(); ++i) {
            double pairwise = estimate_intersection(*sketches[i - 1], *sketches[i]);
            result = std::min(result, pairwise);
        }
        return result;
    }

    // Serialize to raw register bytes for storage in SparseMetadata::hll_sketch
    [[nodiscard]] const std::vector<uint8_t>& serialize() const { return regs_; }

    // Is this sketch empty (no elements added)?
    [[nodiscard]] bool empty() const {
        return std::all_of(regs_.begin(), regs_.end(),
                           [](uint8_t r) { return r == 0; });
    }

private:
    std::vector<uint8_t> regs_;
};

// =============================================================================
// K-Minimum Values (KMV / Theta Sketch)
//
// While HLL is superior for Unions and Cardinalities, KMV naturally supports
// N-way Intersections without the severe inclusion-exclusion numerical errors
// that plague HLL for N >= 3.
// =============================================================================

class KMinValues {
public:
    static constexpr size_t K = 256;

    KMinValues() = default;

    explicit KMinValues(const std::vector<uint64_t>& sketch) {
        for (uint64_t h : sketch) {
            hashes_.insert(h);
            if (hashes_.size() > K) {
                hashes_.erase(std::prev(hashes_.end()));
            }
        }
    }

    void add(uint64_t hash) {
        hashes_.insert(hash);
        if (hashes_.size() > K) {
            hashes_.erase(std::prev(hashes_.end()));
        }
    }

    [[nodiscard]] double estimate() const {
        if (hashes_.empty()) return 0.0;
        if (hashes_.size() < K) return static_cast<double>(hashes_.size());
        
        uint64_t theta = *hashes_.rbegin();
        double max_hash = static_cast<double>(std::numeric_limits<uint64_t>::max());
        return static_cast<double>(K - 1) * max_hash / static_cast<double>(theta);
    }

    [[nodiscard]] const std::set<uint64_t>& get_hashes() const { return hashes_; }

    // Multi-way intersection using the Theta-sketch formulation.
    // It scales the observed exact overlap count by the probability
    // threshold (min_theta) of the space.
    [[nodiscard]] static double estimate_intersection(
        const std::vector<const KMinValues*>& sketches)
    {
        if (sketches.empty()) return 0.0;
        if (sketches.size() == 1) return sketches[0]->estimate();

        uint64_t min_theta = std::numeric_limits<uint64_t>::max();
        for (const auto* sk : sketches) {
            if (sk->get_hashes().empty()) return 0.0; // Intersecting with empty set is 0
            if (sk->get_hashes().size() == K) {
                min_theta = std::min(min_theta, *sk->get_hashes().rbegin());
            }
        }

        int overlap_count = 0;
        for (uint64_t h : sketches[0]->get_hashes()) {
            if (h >= min_theta) break; // we only evaluate in the valid threshold space

            bool in_all = true;
            for (size_t i = 1; i < sketches.size(); ++i) {
                if (sketches[i]->get_hashes().find(h) == sketches[i]->get_hashes().end()) {
                    in_all = false;
                    break;
                }
            }
            if (in_all) overlap_count++;
        }

        if (min_theta == std::numeric_limits<uint64_t>::max()) {
            return static_cast<double>(overlap_count); // Sets were fully exact, no scaling
        }

        double max_hash = static_cast<double>(std::numeric_limits<uint64_t>::max());
        double p = static_cast<double>(min_theta) / max_hash;
        return std::max(1.0, static_cast<double>(overlap_count) / p);
    }

    [[nodiscard]] std::vector<uint64_t> serialize() const {
        return std::vector<uint64_t>(hashes_.begin(), hashes_.end());
    }

private:
    std::set<uint64_t> hashes_;
};

// =============================================================================
// Hash Utilities
// =============================================================================

// MurmurHash3 64-bit finalizer ("mix64"). Fast, good avalanche.
// Use this to hash your domain keys before calling HyperLogLog::add().
inline uint64_t hll_hash(uint64_t key) noexcept {
    key ^= (key >> 33);
    key *= 0xff51afd7ed558ccdULL;
    key ^= (key >> 33);
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= (key >> 33);
    return key;
}

// Combine two hash values (useful for multi-dimensional coordinate hashing)
inline uint64_t hll_hash_combine(uint64_t h1, uint64_t h2) noexcept {
    // Based on boost::hash_combine, adapted for 64 bits
    h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
    return h1;
}

// Hash an Index character for use in mode-level cardinality sketches
inline uint64_t hll_hash_index(char idx, int64_t coord) noexcept {
    return hll_hash(hll_hash_combine(
        static_cast<uint64_t>(static_cast<unsigned char>(idx)),
        static_cast<uint64_t>(coord)));
}
