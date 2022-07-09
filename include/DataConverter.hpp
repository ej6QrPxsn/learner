#ifndef DATA_CONVERTER_HPP
#define DATA_CONVERTER_HPP

#include <torch/torch.h>

using namespace torch::indexing;

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
    policy = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor policy;
};

struct ReplayData {
  ReplayData() {}
  ReplayData(torch::Tensor state_, int batchSize, int seqLength) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, state_.dtype());
    action = torch::empty({batchSize, seqLength, 1}, torch::kUInt8);
    reward = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
    done = torch::empty({batchSize, seqLength, 1}, torch::kBool);
    ih = torch::empty({batchSize, 1, 512}, torch::kFloat32);
    hh = torch::empty({batchSize, 1, 512}, torch::kFloat32);
    policy = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor action;
  torch::Tensor reward;
  torch::Tensor done;
  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor policy;
};

struct Transition : ReplayData {
  Transition() {}
  Transition(torch::Tensor state_, int batchSize, int seqLength, int actionSize)
      : ReplayData(state_, batchSize, seqLength) {
    std::cout << "Transition state " << "state.state: " << state.sizes() << std::endl;
    ih = torch::empty({batchSize, seqLength, 1, 512}, torch::kFloat32);
    hh = torch::empty({batchSize, seqLength, 1, 512}, torch::kFloat32);
    q = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32);
  }

  torch::Tensor ih;
  torch::Tensor hh;
  torch::Tensor q;
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

struct RetraceQ {
  RetraceQ() {}
  RetraceQ(int batchSize, int seqLength, int actionSize) {
    onlineQ = torch::empty({batchSize, seqLength, actionSize}, torch::kFloat32);
  }

  torch::Tensor onlineQ;
};

class DataConverter {
public:
  DataConverter(torch::Tensor &state_, int actionSize_, int seqLength_)
      : state(state_), actionSize(actionSize_), seqLength(seqLength_) {}

  TrainData toBatchedTrainData(std::vector<ReplayData> &dataList);
  RetraceData toBatchedRetraceData(std::vector<ReplayData> &replayDatas, std::vector<RetraceQ> &RetraceQs);

private:
  int actionSize;
  int seqLength;
  torch::Tensor state;
};

#endif // DATA_CONVERTER_HPP
