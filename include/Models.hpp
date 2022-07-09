#ifndef MODELS_HPP
#define MODELS_HPP

#include <regex>
#include <torch/torch.h>
#include <type_traits>

struct Model : torch::nn::Module {
  void saveStateDict(const std::string &file_name) {
    torch::serialize::OutputArchive archive;
    auto params = this->named_parameters(true /*recurse*/);
    auto buffers = this->named_buffers(true /*recurse*/);
    for (const auto &val : params) {
      if (val.value().numel()) {
        archive.write(val.key(), val.value());
      }
    }
    for (const auto &val : buffers) {
      if (val.value().numel()) {
        archive.write(val.key(), val.value(), /*is_buffer*/ true);
      }
    }
    archive.save_to(file_name);
  }

  void loadStateDict(const std::string &file_name,
                     const std::string &ignore_name_regex) {
    torch::serialize::InputArchive archive;
    archive.load_from(file_name);
    torch::NoGradGuard no_grad;
    std::regex re(ignore_name_regex);
    std::smatch m;
    auto params = this->named_parameters(true /*recurse*/);
    auto buffers = this->named_buffers(true /*recurse*/);
    for (auto &val : params) {
      if (!std::regex_match(val.key(), m, re)) {
        archive.read(val.key(), val.value());
      }
    }
    for (auto &val : buffers) {
      if (!std::regex_match(val.key(), m, re)) {
        archive.read(val.key(), val.value(), /*is_buffer*/ true);
      }
    }
  }
};

struct R2D2Agent : Model {
  R2D2Agent(int64_t in_channels, int64_t n_actions) : nActions(n_actions) {
    // (84 - 8) / 4 + 1 = 20
    conv1 = register_module(
        "conv1", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(in_channels, 32, 8).stride(4)));
    // (20 - 4) / 2 + 1 = 9
    conv2 = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2)));
    // (9 - 3) / 1 + 1 = 7
    conv3 = register_module(
        "conv3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)));
    lstm = register_module(
        "lstm", torch::nn::LSTM(
                    torch::nn::LSTMOptions((7 * 7 * 64) + 1 + n_actions, 512)
                        .batch_first(true)));
    adv1 = register_module("adv1", torch::nn::Linear(512, 512));
    adv2 = register_module("adv2", torch::nn::Linear(512, n_actions));
    state1 = register_module("state1", torch::nn::Linear(512, 512));
    state2 = register_module("state2", torch::nn::Linear(512, 1));
  }

  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
  forward(torch::Tensor state, torch::Tensor prevAction,
          torch::Tensor prevReward);
  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
  forward(torch::Tensor state, torch::Tensor prevAction,
          torch::Tensor prevReward, torch::Tensor ih, torch::Tensor hh);

private:
  torch::Tensor forwardConv(torch::Tensor x);
  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
  forwardLstm(torch::Tensor x, torch::Tensor prevAction,
              torch::Tensor prevReward);
  std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
  forwardLstm(torch::Tensor x, torch::Tensor prevAction,
              torch::Tensor prevReward, torch::Tensor ih, torch::Tensor hh);
  torch::Tensor forwardDueling(torch::Tensor x);

  int64_t nActions;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  torch::nn::LSTM lstm{nullptr};
  torch::nn::Linear adv1{nullptr};
  torch::nn::Linear adv2{nullptr};
  torch::nn::Linear state1{nullptr};
  torch::nn::Linear state2{nullptr};
};

#endif // MODELS_HPP
