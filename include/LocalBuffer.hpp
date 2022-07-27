#ifndef LOCAL_BUFFER_HPP
#define LOCAL_BUFFER_HPP

#include "DataConverter.hpp"
#include "Utils.hpp"
#include <memory>
#include <torch/torch.h>

class Request;

struct InferInput {
  InferInput() {}
  InferInput(torch::Tensor state_, int batchSize, int seqLength) {
    auto stateSizes = std::vector<int64_t>{batchSize, seqLength};
    auto stateShape = state_.sizes();
    stateSizes.insert(stateSizes.end(), stateShape.begin(), stateShape.end());

    state = torch::empty(stateSizes, torch::kFloat32);
    prevAction = torch::empty({batchSize, seqLength}, torch::kLong);
    PrevReward = torch::empty({batchSize, seqLength, 1}, torch::kFloat32);
  }

  torch::Tensor state;
  torch::Tensor prevAction;
  torch::Tensor PrevReward;
};

class LocalBuffer {
public:
  LocalBuffer(torch::Tensor &_state, int numEnvs, int _returnSize)
      : returnSize(_returnSize), state(_state),
        transitions(numEnvs, Transition(_state, 1, SEQ_LENGTH, ACTION_SIZE)),
        indexes(numEnvs, 0), prevIndexes(numEnvs, 0),
        replayList(_returnSize, ReplayData(_state, 1, SEQ_LENGTH)),
        qList(_returnSize, RetraceQ(1, SEQ_LENGTH, ACTION_SIZE)),
        prevIh(numEnvs, torch::zeros({1, 512}, torch::kFloat32)),
        prevHh(numEnvs, torch::zeros({1, 512}, torch::kFloat32)) {}

  void setInferenceParam(int envId, Request &request, InferInput *inferData);
  bool update(int envId, Request &request, torch::Tensor &action,
              torch::Tensor &ih, torch::Tensor &hh, torch::Tensor &q,
              torch::Tensor &policy);

  void getTransitions(std::vector<ReplayData> *retReplay,
                      std::vector<RetraceQ> *retRetrace);

private:
  const int returnSize;
  int replaySize = 0;

  torch::Tensor state;
  std::vector<Transition> transitions;
  std::vector<ReplayData> replayList;
  std::vector<RetraceQ> qList;
  std::vector<int> indexes;
  std::vector<int> prevIndexes;
  std::vector<torch::Tensor> prevIh;
  std::vector<torch::Tensor> prevHh;
};

#endif // LOCAL_BUFFER_HPP
