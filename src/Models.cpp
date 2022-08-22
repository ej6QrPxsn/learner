#include "Models.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <torch/script.h>

torch::Dimname dimnameFromString(const std::string &str) {
  return torch::Dimname::fromSymbol(torch::Symbol::dimname(str));
}

using namespace torch::indexing;

AgentOutput R2D2Agent::forward(AgentInput &agentInput) {
  auto x = forwardConv(agentInput.state);
  auto output = forwardLstm(x, agentInput);
  auto batchSize = x.size(0);
  auto seqLen = x.size(1);

  auto input = output.q;

  // batch, seq
  x = forwardDueling(input);

  output.q = x.reshape({batchSize, seqLen, -1});
  return output;
}

torch::Tensor R2D2Agent::forwardConv(torch::Tensor x) {
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

AgentOutput R2D2Agent::forwardLstm(torch::Tensor x, AgentInput &agentInput) {
  auto batchSize = x.size(0);
  auto seqLen = x.size(1);

  // batch, (burn_in + )seq, actions
  auto prevActionOneHot = torch::one_hot(agentInput.prevAction, nActions);

  // batch, (burn_in + )seq, conv outputs + reward + actions
  auto lstmInputs = torch::cat({x, agentInput.prevReward, prevActionOneHot}, 2);

  LstmOutput ret;
  AgentOutput output;
  // バッチごとにlstmの初期状態を設定し、ならし運転をする
  // burn in
  if (agentInput.ih != nullptr && agentInput.hh != nullptr) {
    ret = lstm->forward(lstmInputs,
                         std::make_tuple(agentInput.ih->permute({1, 0, 2}),
                                         agentInput.hh->permute({1, 0, 2})));
  } else {
    ret = lstm->forward(lstmInputs);
  }
  output.q = std::get<0>(ret);
  output.ih = std::get<0>(std::get<1>(ret));
  output.hh = std::get<1>(std::get<1>(ret));
  return output;
}

torch::Tensor R2D2Agent::forwardDueling(torch::Tensor x) {
  auto adv = torch::relu(adv1->forward(x));
  adv = adv2->forward(adv);
  adv -= torch::mean(adv, -1, true);

  auto stv = torch::relu(state1->forward(x));
  stv = state2->forward(stv);

  return adv + stv;
}
