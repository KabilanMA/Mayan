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

### Build and Run


### Notes

`estimate_intersection` is absolutely crucial for estimating matches along shared contracted indices (like the k in matrix multiplication $A(i,k) \times B(k,j)$), that is not the only time it is suitable.

More accurately, HLL intersection is suitable for any operation that acts as a logical `AND` filter, requiring coordinates to be present in both sets to produce a non-zero output. Therefore element-wise multiplication is also suitable but element-wise addition is not suitable and also not suitable for outer product (No shared indices $C(i,j) = A(i) \otimes B(j)$)

__1D Sketches (Mode Sketches)__

A 1D sketch tracks the unique coordinates along a single dimension (or "mode") of a tensor, completely ignoring the other dimensions. It is essentially a projection of the sparse tensor onto one axis.

Imagine a sparse matrix A(i,j) with non-zero values at exactly three locations: (0, 1), (0, 5), and (3, 5). <br>
If you build a 1D sketch for mode i (the rows), you only hash the i values: 0, 0, and 3. The sketch will estimate the cardinality as 2 (because there are non-zeros in only two distinct rows: 0 and 3). <br>
If you build a 1D sketch for mode j (the columns), you only hash the j values: 1, 5, and 5. The sketch will estimate the cardinality as 2 (distinct columns 1 and 5).

---

For Intersections (Element-wise & Contractions): Use KMV only.

Why: These operations depend on estimating the size of an intersection. KMV (Theta Sketches) is mathematically designed for this and handles multi-way intersections accurately. HLL intersection is notoriously inaccurate and numerically unstable.
Logic: If all operands involved in an intersection have KMV sketches (either full or per-mode), use them. If any operand is missing a KMV sketch, we should immediately fall back to a heuristic like the density-product formula, completely avoiding the unreliable HLL intersection.


For Projections & Global Cardinality: Use HLL.

Why: Estimating the number of unique values in a single dimension (projection) or in an entire tensor (global cardinality) is what HLL was designed for. It is extremely fast and space-efficient for this task.
Logic: When we need an operand's total NNZ or the unique count of one of its indices (for a self-reduction), we will use its HLL sketch (hll_sketch or mode_sketches).


### Different Joins for Sparse Tensors NNZ Estimation

1. All involved subset of input operands have the same indices and the output temporary expected from the computation of these operands also has the same indices.
$A(i,j) = B(i,j) * C(i,j) * D(i,j)$. This is an element-wise tensor operation. 
For estimation we expect a non-zero value from all tensors in the same index to produce a non-zero value.

2. There can be intermediate contraction very similar to element-wise operation. This is more generalized version of the above. If we have one or more indices repeating in multiple input and also output tensor, these also should be treated in the same way for NNZ estimation.

2. In any estimation, if one operand have a self-contracting private index ($A(i,j) = B(i,j) * C(i,j) * D(i,k)$, here k is that index), we then have to project that to other dimension before any estimation.

3. If there are one or more shared contracted indices between the involved tensor, which should be contracted away with the temporary after the intermediate tensor fusion, then we should consider N-way contraction to estimate the cost. In this case, our intermediate tensor will directly have N children. But irrespective of it, we need to estimate the cost of that tensor fusion

4. For outer-product of tensors like $A(i,j) = B(i) * C(j)$, the estimated nnz should be equalt to the product of the individual input tensors' estimated nnz.

5. There can be contraction in which there will be one or more index in only one input tensor and present in the intermediate tensor too.


### HyperLogLog

HyperLogLog (HLL) is a probabilistic data structure used to solve the "count-distinct" problem: estimating the number of unique elements (cardinality) in a massive dataset.

Instead of storing every element—which takes O(N) memory—HyperLogLog requires a fixed, incredibly small amount of memory (often just a few kilobytes) to estimate cardinalities in the billions with a typical error rate of around 2%.

__The Coin Flip Analogy__

Imagine flipping a coin and recording the sequence. Observing a sequence of "Heads, Tails" is common. However, observing a sequence of 10 "Heads" in a row is rare. The probability of getting k consecutive heads is $\frac{1}{2^k}$​.

If you tell me the longest streak of consecutive heads you saw was 5, I can reasonably estimate that you flipped the coin roughly $2^5 = 32$ times. HyperLogLog applies this exact probability principle to binary data.

__Hashing for Uniformity__

To apply the coin-flip logic to data, HLL first passes every incoming element through a high-quality hash function (like MurmurHash). This converts arbitrarily sized inputs into uniformly distributed binary strings (e.g., 64-bit integers).

Because the hash function distributes values uniformly, the bits in the hash act like random coin flips.
- The probability of a hash starting with 0 is 50% (1/2).
- The probability of a hash starting with 00 is 25% (1/4).
- The probability of a hash starting with k leading zeros is 1/2k.

By keeping track of the maximum number of leading zeros observed across all hashed elements, we can estimate the total number of unique elements as roughly $2^{max_leading_zeros}$.

__The Variance Problem and Stochastic Averaging__

Relying on a single maximum value is highly volatile. A single "lucky" hash with 20 leading zeros early on would wildly overestimate the cardinality.

To fix this variance, HLL uses a technique called stochastic averaging:
1. Bucketing: It takes the first b bits of the hash to determine a "bucket" or "register" index. The number of buckets is $m = 2^b$.
2. Counting: It takes the remaining bits and counts the number of leading zeros (let's call this value $\rho$).
3. Updating: It updates the register at the chosen index to store the maximum $\rho$ seen for that specific bucket.

If we use 14 bits for the bucket index, we get $2^{14} = 16,384$ separate buckets. We are now running 16,384 independent coin-flip experiments simultaneously, which drastically reduces the variance.

__The Harmonic Mean__

To combine the estimates from all the buckets, we cannot use a simple arithmetic mean. A single outlier bucket with an unusually high number of leading zeros would skew the average upward.

Instead, HyperLogLog uses the harmonic mean to aggregate the estimates. The harmonic mean inherently ignores large outliers and favors smaller, more stable values.

The final cardinality estimate E is calculated using this formula:

$$
E = \alpha _{m} m^2 (\sum\limits_{j=1}^{m} 2^{-M[j]})^{-1}
$$

Where:
- $m$ is the total number of registers (buckets).
- $M[j]$ is the maximum number of leading zeros recorded in register $j$.
- $\alpha _m$​ is a statistically derived constant that corrects systematic multiplicative bias, dependent on the number of registers.

### K-Minimum Values (KMV)

Like HyperLogLog, K-Minimum Values (KMV) is a probabilistic algorithm used to estimate the number of unique elements (cardinality) in a massive dataset but comparatively expensive than HLL.

While HyperLogLog relies on observing rare bit patterns (leading zeros) to guess the scale of the data, KMV relies on the density of uniformly distributed numbers.

__The Core Intuition: The Number Line__

Imagine a continuous number line starting at 0.0 and ending at 1.0.

If you take 3 distinct items, hash them perfectly uniformly, and place them on this line, they will likely divide the line into 4 roughly equal segments. The expected distance between any two points (and between 0 and the first point) is $\frac{1}{3+1} = 0.25$.

If you hash 99 distinct items, they divide the line into 100 segments, and the expected distance between them shrinks to 0.01.

KMV exploits this geometric property: the more distinct items you hash, the closer the smallest hashes get to zero. By measuring exactly how close to zero the k-th smallest hash is, you can mathematically estimate the total number of items.

__Algorithm__
The mechanics of KMV are remarkably simple to implement:

1. Choose $k$: Decide how many minimum values you want to track (e.g., $k=1024$). A larger $k$ requires more memory but yields a more accurate estimate.
2. Initialize a Bounded Heap: Create a max-heap priority queue with a maximum capacity of $k$.
3. Hash and Store: As each element arrives, pass it through a high-quality hash function and normalize the output to a floating-point number between 0 and 1. Let's call this $h$.
- If the heap has fewer than $k$ elements, insert $h$.
- If the heap is full, compare $h$ to the maximum value currently in the heap. If $h$ is smaller, remove the max value and insert $h$. If it's larger, discard it.
4. Calculate: Once all data has been processed, you simply look at the $k$-th smallest value (which is the maximum value sitting at the top of your bounded heap).

Let $U_{(k)}$​ be the $k$-th smallest hash value. The unbiased mathematical estimator for the total number of distinct elements $N$ is:
$$
\hat{N} = \frac{k-1}{U_{(k)}}
$$