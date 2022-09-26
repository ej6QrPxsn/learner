#ifndef UTILS_HPP
#define UTILS_HPP

#include "Models.hpp"
#include "StructuredData.hpp"
#include <torch/torch.h>

std::tuple<float, torch::Tensor>
retraceLoss(const torch::Tensor action, const torch::Tensor reward,
            const torch::Tensor done, const torch::Tensor policy,
            const torch::Tensor onlineQ, const torch::Tensor targetQ,
            const torch::Device device, bool backward = false);

StoredData compress(ReplayData &replayData);
void decompress(StoredData &compressed, ReplayData &replayData);
void toBatchedTrainData(TrainData &train,
                        std::array<ReplayData, BATCH_SIZE> &dataList);

#endif // UTILS_HPP
