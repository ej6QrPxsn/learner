#include "Models.hpp"
#include <iostream>
#include <torch/script.h>
#include <array>
#include <iostream>
#include <iomanip>

torch::Dimname dimnameFromString(const std::string &str)
{
  return torch::Dimname::fromSymbol(torch::Symbol::dimname(str));
}

using namespace torch::indexing;

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> R2D2Agent::forward(torch::Tensor state, torch::Tensor prevAction, torch::Tensor prevReward)
{
  auto x = forwardConv(state);
  auto lstmRet = forwardLstm(x, prevAction, prevReward);
  auto batchSize = state.size(0);
  auto seqLen = state.size(1);

  auto input = std::get<0>(lstmRet);
  auto lstm_state = std::get<1>(lstmRet);

  // batch, seq
  x = forwardDueling(input);
  return std::make_tuple(x.reshape({batchSize, seqLen, -1}), lstm_state);
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> R2D2Agent::forward(torch::Tensor state, torch::Tensor prevAction, torch::Tensor prevReward, torch::Tensor ih, torch::Tensor hh)
{
  auto x = forwardConv(state);
  auto lstmRet = forwardLstm(x, prevAction, prevReward, ih, hh);
  auto batchSize = state.size(0);
  auto seqLen = state.size(1);

  auto input = std::get<0>(lstmRet);
  auto lstm_state = std::get<1>(lstmRet);

  // batch, seq
  x = forwardDueling(input);
  return std::make_tuple(x.reshape({batchSize, seqLen, -1}), lstm_state);
}

torch::Tensor R2D2Agent::forwardConv(torch::Tensor x)
{
  auto batchSize = x.size(0);
  auto seqLen = x.size(1);

  // batch * seq, channel, w, h
  x = x.reshape({-1, x.sizes()[2], x.sizes()[3], x.sizes()[4]});

  x = torch::relu(conv1->forward(x));
  x = torch::relu(conv2->forward(x));
  x = torch::relu(conv3->forward(x));

  // batch, seq, conv outputs
  return x.reshape({batchSize, seqLen, -1});
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
R2D2Agent::forwardLstm(torch::Tensor x, torch::Tensor prevAction, torch::Tensor prevReward, torch::Tensor ih, torch::Tensor hh)
{
  auto batchSize = x.size(0);
  auto seqLen = x.size(1);

  // batch, (burn_in + )seq, actions
  auto prevActionOneHot = torch::one_hot(prevAction, nActions);

  // std::cout << "x: " << x.sizes() << std::endl;
  // std::cout << "prevReward: " << prevReward.sizes() << std::endl;
  // std::cout << "prevAction_one_hot: " << prevAction_one_hot.sizes() << std::endl;

  // std::cout << "lstm x " << x.sizes() << std::endl;
  // std::cout << "lstm prevReward " << prevReward.sizes() << std::endl;
  // std::cout << "lstm prevActionOneHot " << prevActionOneHot.sizes() << std::endl;

  // batch, (burn_in + )seq, conv outputs + reward + actions
  auto lstmInputs = torch::cat({x, prevReward, prevActionOneHot}, 2);

  // std::cout << "lstm_inputs: " << lstm_inputs.sizes() << std::endl;

  // バッチごとにlstmの初期状態を設定し、ならし運転をする
  // burn in
  return lstm->forward(lstmInputs, std::make_tuple(ih.permute({1, 0, 2}), hh.permute({1, 0, 2})));
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
R2D2Agent::forwardLstm(torch::Tensor x, torch::Tensor prevAction, torch::Tensor prevReward)
{
  auto batchSize = x.sizes()[0];
  auto seqLen = x.sizes()[1];

  // batch, (burn_in + )seq, actions
  auto prevActionOneHot = torch::one_hot(prevAction, nActions);

  // std::cout << "x: " << x.sizes() << std::endl;
  // std::cout << "prevReward: " << prevReward.sizes() << std::endl;
  // std::cout << "prevAction_one_hot: " << prevAction_one_hot.sizes() << std::endl;

  // batch, (burn_in + )seq, conv outputs + reward + actions
  auto lstmInputs = torch::cat({x, prevReward, prevActionOneHot}, 2);

  // std::cout << "lstm_inputs: " << lstm_inputs.sizes() << std::endl;

  // Q値取得
  return lstm->forward(lstmInputs);
}

torch::Tensor R2D2Agent::forwardDueling(torch::Tensor x)
{
  auto adv = torch::relu(adv1->forward(x));
  adv = adv2->forward(adv);
  adv -= torch::mean(adv, -1, true);

  auto stv = torch::relu(state1->forward(x));
  stv = state2->forward(stv);

  return adv + stv;
}

