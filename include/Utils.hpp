#ifndef UTILS_HPP
#define UTILS_HPP

#include "StructuredData.hpp"
#include <torch/torch.h>

std::tuple<float, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ,
            torch::optim::Adam *optimizer = nullptr);

StoredData compress(ReplayData &replayData);
void decompress(StoredData &compressed, ReplayData &replayData);

#endif // UTILS_HPP
