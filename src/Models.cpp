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
  auto batchSize = state.sizes()[0];
  auto seqLen = state.sizes()[1];

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

  // batch, (burn_in + )seq, conv outputs + reward + actions
  std::cout << "lstm x " << x.sizes() << std::endl;
  std::cout << "lstm prevReward " << prevReward.sizes() << std::endl;
  std::cout << "lstm prevActionOneHot " << prevActionOneHot.sizes() << std::endl;

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

// int main(void)
// {
//   int batchSize = 32;
//   int SEQ_SIZE = 120;
//   int ACTION = 9;

//   // Create a new Net.
//   auto net = std::make_shared<R2D2Agent>(1, ACTION);
//   // auto forward(torch::Tensor x, torch::Tensor prevAction, torch::Tensor prevReward, torch::Tensor ih, torch::Tensor hh, bool burn_in)

//   auto state = torch::randint(255, {batchSize, SEQ_SIZE, 1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
//   auto prevAction = torch::randint(ACTION, {batchSize, SEQ_SIZE}, torch::TensorOptions().dtype(torch::kUInt8));
//   auto prevReward = torch::rand({batchSize, SEQ_SIZE, 1}, torch::TensorOptions().dtype(torch::kFloat32));
//   auto ih = torch::rand({batchSize, SEQ_SIZE, 1, 512}, torch::TensorOptions().dtype(torch::kFloat32));
//   auto hh = torch::rand({batchSize, SEQ_SIZE, 1, 512}, torch::TensorOptions().dtype(torch::kFloat32));
//   bool burn_in = true;

//   // std::cout << "state: " << state << std::endl;
//   // std::cout << "prevAction: " << prevAction << std::endl;
//   // std::cout << "prevRewardte: " << prevReward << std::endl;
//   // std::cout << "ih: " << ih.index({Slice (), 1}).sizes() << std::endl;
//   // std::cout << "hh: " << hh.index({Slice (), 1}).sizes() << std::endl;

//   // Instantiate an SGD optimization algorithm to update our Net's parameters.
//   torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

//   auto ret = net->forward(state / 255.0, prevAction.to(torch::kLong), prevReward,
//   ih.index({Slice (), 1}),
//   hh.index({Slice (), 1}),
//   burn_in);
//   auto losses = std::get<0>(ret);
//   auto lstm_states = std::get<1>(ret);
//   auto in_ih = std::get<0>(lstm_states);
//   auto in_hh = std::get<1>(lstm_states);

//   // std::cout << "in_q: " << in_q << std::endl;
//   // std::cout << "in_ih: " << in_ih << std::endl;
//   // std::cout << "in_hh: " << in_hh << std::endl;

//   auto loss = torch::sum(losses);
//   // Compute gradients of the loss w.r.t. the parameters of our model.
//   loss.backward();
//   // Update the parameters based on the calculated gradients.
//   optimizer.step();

//   int a = 1;
// }
