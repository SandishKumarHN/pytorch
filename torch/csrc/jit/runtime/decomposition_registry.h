#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Map to track original symbolic size nodes and their corresponding values
using SymSizeMap = std::unordered_map<Value*, Value*>;

TORCH_API std::optional<std::shared_ptr<Graph>> GetDecomposition(
    const FunctionSchema& schema);

TORCH_API void RegisterDecomposition(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g);

TORCH_API void RunDecompositions(std::shared_ptr<Graph> g);

TORCH_API std::optional<GraphFunction*> GetDecompositionFunction(
    const FunctionSchema& schema);

// For invocation in C++, recommended is to assign to static local variable
TORCH_API Function* GetDecompositionExecutor(const char* schema_literal);

TORCH_API Function* GetDecompositionExecutor(const FunctionSchema& schema);

TORCH_API void run_jit_decomposition(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

TORCH_API bool has_jit_decomposition(const FunctionSchema& schema);

// TORCH_API void RestoreSymbolicSizes(std::shared_ptr<Graph>& graph, const SymSizeMap& symSizeMap);

// TORCH_API void RunDecompositionsWithSymSizeTracking(Block* block, SymSizeMap& symSizeMap);

} // namespace torch::jit
