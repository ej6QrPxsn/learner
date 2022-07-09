#ifndef UTILS_HPP
#define UTILS_HPP

#include <torch/torch.h>

const auto RETRACE_LAMBDA = 0.95;
const auto RESCALING_EPSILON = 1e-3;
const auto ETA = 0.9;
const auto ACTION_SIZE = 9;
const auto DISCOUNT_GAMMA = 0.997;

std::tuple<torch::Tensor, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ);

#endif // UTILS_HPP
