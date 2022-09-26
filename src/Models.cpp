#include "Models.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <torch/script.h>

torch::Dimname dimnameFromString(const std::string &str) {
  return torch::Dimname::fromSymbol(torch::Symbol::dimname(str));
}

using namespace torch::indexing;
std::mutex mtx;

AgentOutput R2D2Agent::forward(const torch::Tensor x,
                               const torch::Tensor prevAction,
                               const torch::Tensor prevReward,
                               const LstmStates initialLstmStates,
                               const torch::Device device) {
  auto batchSize = x.size(0);
  auto seqLen = x.size(1);

  torch::Tensor out, feature, feature1, feature2;
  auto [prevHiddenState, prevCellState] = initialLstmStates;
  // auto lstmStatesStack =
  //     torch::empty({batchSize, seqLen, LSTM_STATE_SIZE}).to(device);

  // batch * seq, channel, w, h
  feature = x.contiguous().view({-1, x.sizes()[2], x.sizes()[3], x.sizes()[4]});
  feature = conv1->forward(feature);
  feature = torch::relu(feature);
  feature = conv2->forward(feature);
  feature = torch::relu(feature);
  feature = conv3->forward(feature);
  feature = torch::relu(feature);

  feature = feature.contiguous().view({batchSize, seqLen, -1});

  // batch, (burn_in + )seq, actions
  auto prevActionOneHot = torch::one_hot(prevAction, nActions);

  // batch, (burn_in + )seq, conv outputs + reward + actions
  auto lstmInputs = torch::cat({feature, prevReward, prevActionOneHot}, 2);

  std::vector<torch::Tensor> lstmOutputs(
      seqLen, torch::empty({batchSize, LSTM_STATE_SIZE}).to(device));
  for (long i = 0; i < seqLen; i++) {
    auto [newHiddenState, newCellState] =
        lstmCell(lstmInputs.index({Slice(), i, Slice()}),
                 std::make_tuple(prevHiddenState, prevCellState));
    lstmOutputs[i] = newCellState;
    prevHiddenState = newHiddenState;
    prevCellState = newCellState;
  }
  auto lstmStatesStack =
      torch::stack(lstmOutputs, 0).permute({1, 0, 2}).to(device);

  // auto [hiddenState, nextCellState] = prevLstmStates;

  // auto lstmStatesStack = fc1->forward(lstmInputs);

  feature1 = adv1->forward(lstmStatesStack);
  feature1 = torch::relu(feature1);
  feature1 = adv2->forward(feature1);
  feature1 -= torch::mean(feature1, -1, true);

  feature2 = state1->forward(lstmStatesStack);
  feature2 = torch::relu(feature2);
  feature2 = state2->forward(feature2);

  out = feature1 + feature2;

  // batch, seq, conv outputs
  return {out, std::make_tuple(prevHiddenState, prevCellState)};
}
