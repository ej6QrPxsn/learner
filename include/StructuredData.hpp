#ifndef STRUCTURED_DATA_HPP
#define STRUCTURED_DATA_HPP

#include "Common.hpp"
#include <torch/torch.h>

struct Request {
  int envId;
  uint8_t state[STATE_SIZE];
  float reward;
  bool done;
} __attribute__((packed));

struct AgentInput {
  AgentInput() {}
  AgentInput(torch::Tensor state_, int batchSize, int seqLength,
             torch::Device device) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32).to(device);
    prevAction = torch::empty({batchSize, seqLength}, torch::kLong).to(device);
    prevReward =
        torch::empty({batchSize, seqLength, 1}, torch::kFloat32).to(device);
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

struct ReplayData {
  ReplayData() {}
  ReplayData & getReplayData() {
    return *this;
  }

  uint8_t state[SEQ_LENGTH][STATE_SIZE];
  uint8_t action[SEQ_LENGTH];
  float reward[SEQ_LENGTH];
  float policy[SEQ_LENGTH];
  float ih[LSTM_STATE_SIZE];
  float hh[LSTM_STATE_SIZE];
  bool done[SEQ_LENGTH];
};

struct Transition : ReplayData {
  Transition() {}

  ReplayData & getReplayData() {
    std::copy(ih[1], ih[1] + LSTM_STATE_SIZE, ReplayData::ih);
    std::copy(hh[1], hh[1] + LSTM_STATE_SIZE, ReplayData::hh);

    return ReplayData::getReplayData();
  }

  float ih[SEQ_LENGTH][LSTM_STATE_SIZE];
  float hh[SEQ_LENGTH][LSTM_STATE_SIZE];
  float q[SEQ_LENGTH][ACTION_SIZE];
};

struct RetraceData {
  RetraceData() {}
  RetraceData(int batchSize, int seqLength, int actionSize,
              torch::Device device) {

    // auto options = torch::TensorOptions()
    //                    .dtype(torch::kFloat32)
    //                    .layout(torch::kStrided)
    //                    .device(torch::kCUDA, 1)
    //                    .requires_grad(true);

    action = torch::empty({batchSize, seqLength, 1}, torch::kInt64).to(device);
    reward = torch::empty({batchSize, seqLength}, torch::kFloat32).to(device);
    done = torch::empty({batchSize, seqLength}, torch::kBool).to(device);
    policy = torch::empty({batchSize, seqLength}, torch::kFloat32).to(device);
    onlineQ = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32)
                  .to(device);
    targetQ = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32)
                  .to(device);
  }

  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor policy;
  torch::Tensor onlineQ;
  torch::Tensor targetQ;
};

struct StoredData {
  int size = 0;
  float reward = 0;
  std::unique_ptr<char[]> ptr;
};

struct SampleData {
  std::array<ReplayData, BATCH_SIZE> dataList;
  std::array<int, BATCH_SIZE> indexList;
  std::array<int, BATCH_SIZE> labelList;
};

struct TrainData {
  TrainData() {}
  TrainData(torch::Tensor state_, int batchSize, int seqLength,
            torch::Device device) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32).to(device);
    action = torch::empty({batchSize, seqLength}, torch::kLong).to(device);
    reward =
        torch::empty({batchSize, seqLength, 1}, torch::kFloat32).to(device);
    done = torch::empty({batchSize, seqLength, 1}, torch::kBool).to(device);
    ih = torch::empty({batchSize, 1, 512}, torch::kFloat32).to(device);
    hh = torch::empty({batchSize, 1, 512}, torch::kFloat32).to(device);
    policy = torch::empty({batchSize, seqLength}, torch::kFloat32).to(device);
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
