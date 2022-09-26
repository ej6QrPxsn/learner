#ifndef MODELS_HPP
#define MODELS_HPP

#include "StructuredData.hpp"
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

  void copyParams(NamedParameters &newParams, NamedParameters &newBuffers) {
    torch::NoGradGuard no_grad;

    auto params = this->named_parameters(true /*recurse*/);
    for (auto &val : newParams) {
      auto name = val.key();
      auto *t = params.find(name);
      if (t != nullptr) {
        t->copy_(val.value());
      }
    }

    auto buffers = this->named_buffers(true /*recurse*/);
    for (auto &val : newBuffers) {
      auto name = val.key();
      auto *t = buffers.find(name);
      if (t != nullptr) {
        t->copy_(val.value());
      }
    }
  }

  void copyFrom(Model &fromModel) {
    auto newParams = fromModel.named_parameters(true /*recurse*/);
    auto newBuffers = fromModel.named_parameters(true /*recurse*/);
    copyParams(newParams, newBuffers);
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

    lstmCell = register_module(
        "lstm",
        torch::nn::LSTMCell((7 * 7 * 64) + 1 + n_actions, LSTM_STATE_SIZE));

    adv1 = register_module("adv1", torch::nn::Linear(LSTM_STATE_SIZE, 512));
    adv2 = register_module("adv2", torch::nn::Linear(512, n_actions));
    state1 = register_module("state1", torch::nn::Linear(LSTM_STATE_SIZE, 512));
    state2 = register_module("state2", torch::nn::Linear(512, 1));
    // fc1 = register_module("fc1", torch::nn::Linear((7 * 7 * 64) + 1 +
    // n_actions,
    //                                                LSTM_STATE_SIZE));
  }

  AgentOutput forward(const torch::Tensor x, const torch::Tensor prevAction,
                      const torch::Tensor prevReward,
                      const LstmStates lstmStates, const torch::Device device);

  void detach_() {
    conv1->weight.detach_();
    conv2->weight.detach_();
    conv3->weight.detach_();
    lstmCell->weight_hh.detach_();
    lstmCell->weight_hh.detach_();
    adv1->weight.detach_();
    adv2->weight.detach_();
    state1->weight.detach_();
    state2->weight.detach_();
    conv1->bias.detach_();
    conv2->bias.detach_();
    conv3->bias.detach_();
    lstmCell->bias_ih.detach_();
    lstmCell->bias_hh.detach_();
    adv1->bias.detach_();
    adv2->bias.detach_();
    state1->bias.detach_();
    state2->bias.detach_();
  }

  void requiresGrad_(bool grad) {
    conv1->weight.requires_grad_(grad);
    conv2->weight.requires_grad_(grad);
    conv3->weight.requires_grad_(grad);

    lstmCell->weight_hh.requires_grad_(grad);
    lstmCell->weight_hh.requires_grad_(grad);

    adv1->weight.requires_grad_(grad);
    adv2->weight.requires_grad_(grad);
    state1->weight.requires_grad_(grad);
    state2->weight.requires_grad_(grad);

    conv1->bias.requires_grad_(grad);
    conv2->bias.requires_grad_(grad);
    conv3->bias.requires_grad_(grad);

    lstmCell->bias_ih.requires_grad_(grad);
    lstmCell->bias_hh.requires_grad_(grad);
    adv1->bias.requires_grad_(grad);
    adv2->bias.requires_grad_(grad);
    state1->bias.requires_grad_(grad);
    state2->bias.requires_grad_(grad);
  }

  // private:
  int64_t nActions;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  // torch::nn::Linear fc1{nullptr};
  torch::nn::LSTMCell lstmCell{nullptr};
  torch::nn::Linear adv1{nullptr};
  torch::nn::Linear adv2{nullptr};
  torch::nn::Linear state1{nullptr};
  torch::nn::Linear state2{nullptr};
};

#endif // MODELS_HPP
