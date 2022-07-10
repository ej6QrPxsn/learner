#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>
#include "Common.hpp"

std::tuple<torch::Tensor, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ);

#endif // UTILS_HPP
