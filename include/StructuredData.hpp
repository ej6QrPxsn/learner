#ifndef STRUCTURED_DATA_HPP
#define STRUCTURED_DATA_HPP

#include "Common.hpp"
#include <torch/torch.h>

struct Request {
  Request() {}
  Request(torch::Tensor state_, float reward_, bool done_)
      : state(state_), reward(reward_), done(done_) {}
  torch::Tensor state;
  float reward;
  bool done;
};

struct AgentInput {
  AgentInput() {}
  AgentInput(torch::Tensor state_, int batchSize, int seqLength) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32);
    prevAction = torch::empty({batchSize, seqLength}, torch::kLong);
    prevReward = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor prevAction;
  torch::Tensor prevReward;
  torch::Tensor *ih = nullptr;
  torch::Tensor *hh = nullptr;
};

struct AgentOutput {
  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor q;
};

struct Transition {
  Transition() {}
  Transition(torch::Tensor state_, int batchSize, int seqLength,
             int actionSize) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, state_.dtype());
    action = torch::empty({batchSize, seqLength, 1}, torch::kUInt8);
    reward = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
    done = torch::empty({batchSize, seqLength, 1}, torch::kBool);
    policy = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);

    ih = torch::empty({batchSize, seqLength, 512}, torch::kFloat32);
    hh = torch::empty({batchSize, seqLength, 512}, torch::kFloat32);
    q = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor policy;
  torch::Tensor q;
};

struct ReplayData {
  ReplayData() {}
  uint8_t state[SEQ_LENGTH * STATE_SIZE];
  uint8_t action[SEQ_LENGTH];
  float reward[SEQ_LENGTH];
  bool done[SEQ_LENGTH];
  float ih[LSTM_STATE_SIZE];
  float hh[LSTM_STATE_SIZE];
  float policy[SEQ_LENGTH];
};

struct RetraceQ {
  RetraceQ() {}
  float onlineQ[SEQ_LENGTH * ACTION_SIZE];
};

struct RetraceData {
  RetraceData() {}
  RetraceData(int batchSize, int seqLength, int actionSize) {
    action = torch::empty({batchSize, seqLength, 1}, torch::kInt64);
    reward = torch::empty({batchSize, seqLength}, torch::kFloat32);
    done = torch::empty({batchSize, seqLength}, torch::kBool);
    policy = torch::empty({batchSize, seqLength}, torch::kFloat32);
    onlineQ = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32);
    targetQ = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32);
  }

  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor policy;
  torch::Tensor onlineQ;
  torch::Tensor targetQ;
};

struct StoredData {
  int size;
  std::unique_ptr<char[]> ptr;
};

struct SampleData {
  std::array<ReplayData, BATCH_SIZE> dataList;
  std::array<int, BATCH_SIZE> indexList;
  std::array<int, BATCH_SIZE> labelList;
};

struct TrainData {
  TrainData() {}
  TrainData(torch::Tensor state_, int batchSize, int seqLength) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32);
    action = torch::empty({batchSize, seqLength}, torch::kLong);
    reward = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
    done = torch::empty({batchSize, seqLength, 1}, torch::kBool);
    ih = torch::empty({batchSize, 1, 512}, torch::kFloat32);
    hh = torch::empty({batchSize, 1, 512}, torch::kFloat32);
    policy = torch::empty({batchSize, seqLength}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor policy;
};

#endif // STRUCTURED_DATA_HPP
