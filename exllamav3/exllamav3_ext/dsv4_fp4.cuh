#pragma once
#include <torch/extension.h>
#include <vector>
class Graph;

std::vector<at::Tensor> dsv4_fp4_quant(const at::Tensor& input, bool indexer);
at::Tensor dsv4_fp4_dequant(const at::Tensor& packed, const at::Tensor& scales, bool indexer);
void dsv4_fp4_scatter(const at::Tensor& stage, at::Tensor packed, at::Tensor scales,
    const at::Tensor& block_table, int position, const c10::optional<at::Tensor>& positions,
    int m, int seq, int epp, bool indexer);
void dsv4_fp4_scatter_gr(const at::Tensor& stage, at::Tensor packed, at::Tensor scales,
    const at::Tensor& block_table, int position, const c10::optional<at::Tensor>& positions,
    int m, int seq, int epp, bool indexer, Graph* graph);
void dsv4_fp4_positions(const at::Tensor& positions, at::Tensor output, int m, Graph* graph);
