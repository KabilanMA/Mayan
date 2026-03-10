#include "dp_optimizer.h"
#include <iostream>

int main() {
    std::cout << "==========================================\n";
    std::cout << "   TACO Einsum Frontend Compiler\n";
    std::cout << "==========================================\n\n";

    StorageFormat csr_ij{{LevelType::DENSE, LevelType::COMPRESSED}, {'i', 'j'}};
    StorageFormat csc_jk{{LevelType::COMPRESSED, LevelType::COMPRESSED}, {'k', 'j'}};
    StorageFormat csr_kl{{LevelType::DENSE, LevelType::COMPRESSED}, {'k', 'l'}};

    // A is dense (10,000 nnz). B and C are highly sparse (50 nnz each)
    auto A = std::make_shared<TensorNode>("A", std::vector<Index>{'i', 'j'}, Shape{100, 100}, 50, csr_ij);
    auto B = std::make_shared<TensorNode>("B", std::vector<Index>{'j', 'k'}, Shape{100, 100}, 50, csc_jk);
    auto C = std::make_shared<TensorNode>("C", std::vector<Index>{'k', 'l'}, Shape{100, 100}, 10000, csr_kl);

    std::vector<std::shared_ptr<ExprNode>> inputs = {A, B, C};
    std::vector<Index> global_output = {'i', 'l'};

    std::cout << "Input Equation: A_ij * B_jk * C_kl -> Out_il\n";
    std::cout << "  Tensor A: " << A->to_string() << " (NNZ: " << A->nnz << ")\n";
    std::cout << "  Tensor B: " << B->to_string() << " (NNZ: " << B->nnz << ")\n";
    std::cout << "  Tensor C: " << C->to_string() << " (NNZ: " << C->nnz << ")\n\n";

    std::cout << "Compiling optimal execution path...\n\n";
    auto optimized_ast = DPOptimizer::optimize(inputs, global_output, {{'i', 100}, {'j', 100}, {'k', 100}});

    std::cout << "Optimal Execution Plan:\n";
    std::cout << optimized_ast->to_string() << "\n\n";

    return 0;
}