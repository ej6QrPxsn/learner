#ifndef LOCAL_BUFFER_HPP
#define LOCAL_BUFFER_HPP

#include "StructuredData.hpp"
#include <memory>
#include <torch/torch.h>

class LocalBuffer {
public:
  LocalBuffer(torch::Tensor state_, int numEnvs, torch::Device device_)
      : stateShape(state_.sizes()), device(device_),
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
                              torch::Tensor &q, AgentOutput &agentOutput,
                              torch::Tensor &policy);

private:
  torch::Device device;
  c10::IntArrayRef stateShape;
  Transition transition;
  int index = 0;
  int retraceIndex = 0;
  uint8_t prevAction;
  float prevReward;
  float prevIh[LSTM_STATE_SIZE];
  float prevHh[LSTM_STATE_SIZE];
  RetraceData retraceData;
  std::vector<StoredData> storedDatas;
};

#endif // LOCAL_BUFFER_HPP
