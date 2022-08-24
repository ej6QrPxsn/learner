#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>
#include "Common.hpp"
#include "SumTree.hpp"

struct ReplayData;

std::tuple<torch::Tensor, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ);

StoredData compress(ReplayData & replayData);
ReplayData decompress(StoredData & storedData);

#endif // UTILS_HPP
