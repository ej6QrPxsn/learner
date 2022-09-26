#ifndef LOCAL_BUFFER_HPP
#define LOCAL_BUFFER_HPP

#include "StructuredData.hpp"
#include <memory>
#include <torch/torch.h>

class LocalBuffer {
public:
  LocalBuffer(torch::Tensor state_, int numEnvs, torch::Device device_)
      : stateShape(state_.sizes()), device(device_),
        prevHiddenStates(torch::zeros({1, LSTM_STATE_SIZE})),
        prevCellStates(torch::zeros({1, LSTM_STATE_SIZE})),
        retraceData(BATCH_SIZE, 1 + TRACE_LENGTH, ACTION_SIZE, device_) {}

  RetraceData &getRetraceData() { return retraceData; }
  std::vector<StoredData> getReplayData() {
    std::vector<StoredData> ret;
    ret.swap(storedDatas);
    retraceIndex = 0;
    return ret;
  }

  void setInferenceParam(Request &request, AgentInput *inferData);
  void inline setRetaceData();
  bool updateAndGetTransition(Request &request, torch::Tensor &action,
                              torch::Tensor &q, LstmStates &lstmStates,
                              torch::Tensor &policy);

private:
  torch::Device device;
  int prevAction;
  c10::IntArrayRef stateShape;
  Transition transition;

  int index = 0;
  int retraceIndex = 0;
  float prevReward;
  torch::Tensor prevHiddenStates;
  torch::Tensor prevCellStates;
  RetraceData retraceData;
  std::vector<StoredData> storedDatas;
};

#endif // LOCAL_BUFFER_HPP
