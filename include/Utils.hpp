#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>
#include "StructuredData.hpp"

std::tuple<float, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ,
            torch::Device device, torch::optim::Adam *optimizer = nullptr);

StoredData compress(ReplayData &replayData);
void decompress(StoredData &compressed, ReplayData &replayData);

#endif // UTILS_HPP
