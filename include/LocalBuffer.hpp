#ifndef LOCAL_BUFFER_HPP
#define LOCAL_BUFFER_HPP

#include "DataConverter.hpp"
#include "Utils.hpp"
#include <memory>
#include <torch/torch.h>

class Request;

struct InferenceData {
  InferenceData() {}
  InferenceData(torch::Tensor state_, int batchSize, int seqLength) {
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
  LocalBuffer(torch::Tensor &_state, int _actionSize, int numEnvs,
              int _traceLength, int _replayPeriod, int _returnSize)
      : replayPeriod(_replayPeriod), actionSize(_actionSize),
        seqLength(1 + replayPeriod + _traceLength), returnSize(_returnSize),
        state(_state),
        transitions(numEnvs, Transition(_state, 1, seqLength, actionSize)),
        indexes(numEnvs, 0),
        prevIndexes(numEnvs, 0),
        prevIh(numEnvs, torch::zeros({1, 512}, torch::kFloat32)),
        prevHh(numEnvs, torch::zeros({1, 512}, torch::kFloat32)) {}

  void setInferenceParam(std::vector<int> &envIds, InferenceData *inferData,
                         std::vector<Request> &requests);
  void
  updateAndGetTransitions(std::vector<int> &envIds,
                          std::vector<Request> &requests, torch::Tensor &action,
                          std::tuple<torch::Tensor, torch::Tensor> &lstmStates,
                          torch::Tensor &q, torch::Tensor &policy,
                          std::vector<ReplayData> *retReplay,
                          std::vector<RetraceQ> *retRetrace);

private:
  const int replayPeriod;
  const int actionSize;
  const int seqLength;
  const int returnSize;

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
