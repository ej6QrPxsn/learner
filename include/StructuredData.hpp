#ifndef STRUCTURED_DATA_HPP
#define STRUCTURED_DATA_HPP

#include "Common.hpp"
#include <torch/torch.h>

using NamedParameters = torch::OrderedDict<std::string, at::Tensor>;

// struct LstmStates {
//   LstmStates() {}
//   LstmStates(int batch, torch::Device device)
//       : hiddeState(torch::empty({batch, LSTM_STATE_SIZE}).to(device)),
//         cellState(torch::empty({batch, LSTM_STATE_SIZE}).to(device)) {}
//   LstmStates(torch::Tensor hs, torch::Tensor cs)
//       : hiddeState(hs), cellState(cs) {}
//   torch::Tensor hiddeState;
//   torch::Tensor cellState;
// };

using LstmStates = std::tuple<torch::Tensor, torch::Tensor>;
using AgentOutput = std::tuple<torch::Tensor, LstmStates>;

struct Event {
  bool notify = false;
  std::mutex mtx;
  std::condition_variable cv;

  void wait() {
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&] { return notify; });
      notify = false;
    }
  }

  void set() {
    // 共有データの更新
    notify = true;
    cv.notify_all();
  }

  void reset() {
    std::lock_guard<std::mutex> lk(mtx);
    // 共有データの更新
    notify = false;
  }
};

struct Request {
  int envId;
  uint8_t state[STATE_SIZE];
  float reward;
  bool done;
} __attribute__((packed));

struct AgentInput {
  AgentInput(torch::Tensor state_, int batchSize, int seqLength,
             torch::Device device) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32).to(device);
    prevAction = torch::empty({batchSize, seqLength}, torch::kLong).to(device);
    prevReward =
        torch::empty({batchSize, seqLength, 1}, torch::kFloat32).to(device);
    hiddenStates =
        torch::empty({batchSize, LSTM_STATE_SIZE}, torch::kFloat32).to(device);
    cellStates =
        torch::empty({batchSize, LSTM_STATE_SIZE}, torch::kFloat32).to(device);
  }
  AgentInput(torch::Tensor state_, torch::Tensor prevAction_,
             torch::Tensor prevReward_, torch::Tensor hiddenStates_,
             torch::Tensor cellStates_)
      : state(state_), prevAction(prevAction_), prevReward(prevReward_),
        hiddenStates(hiddenStates_), cellStates(cellStates_) {}

  torch::Tensor state;
  torch::Tensor prevAction;
  torch::Tensor prevReward;
  torch::Tensor hiddenStates;
  torch::Tensor cellStates;
};

struct ReplayData {
  ReplayData() {}
  ReplayData &getReplayData() { return *this; }

  uint8_t state[SEQ_LENGTH][STATE_SIZE];
  uint8_t action[SEQ_LENGTH];
  float reward[SEQ_LENGTH];
  float policy[SEQ_LENGTH];
  float hiddenStates[LSTM_STATE_SIZE];
  float cellStates[LSTM_STATE_SIZE];
  bool done[SEQ_LENGTH];
};

struct Transition : ReplayData {
  Transition() {}

  ReplayData &getReplayData() {
    std::copy(hiddenStates[1], hiddenStates[1] + LSTM_STATE_SIZE,
              ReplayData::hiddenStates);
    std::copy(cellStates[1], cellStates[1] + LSTM_STATE_SIZE,
              ReplayData::cellStates);

    return ReplayData::getReplayData();
  }

  float hiddenStates[SEQ_LENGTH][LSTM_STATE_SIZE];
  float cellStates[SEQ_LENGTH][LSTM_STATE_SIZE];
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
  TrainData() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);

    state = torch::empty({BATCH_SIZE, SEQ_LENGTH, 1, 84, 84}, torch::kFloat32)
                .to(device);
    action = torch::empty({BATCH_SIZE, SEQ_LENGTH}, torch::kLong).to(device);
    reward =
        torch::empty({BATCH_SIZE, SEQ_LENGTH, 1}, torch::kFloat32).to(device);
    done = torch::empty({BATCH_SIZE, SEQ_LENGTH, 1}, torch::kBool).to(device);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       //  .layout(torch::kStrided)
                       //  .device(torch::kCUDA, 1)
                       .requires_grad(true);

    hiddenStates =
        torch::empty({BATCH_SIZE, LSTM_STATE_SIZE}, options).to(device);
    cellStates =
        torch::empty({BATCH_SIZE, LSTM_STATE_SIZE}, options).to(device);
    policy = torch::empty({BATCH_SIZE, SEQ_LENGTH}, torch::kFloat32).to(device);
  }

  torch::Tensor state;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor hiddenStates;
  torch::Tensor cellStates;
  torch::Tensor policy;
};

#endif // STRUCTURED_DATA_HPP
