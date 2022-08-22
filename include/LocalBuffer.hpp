#ifndef LOCAL_BUFFER_HPP
#define LOCAL_BUFFER_HPP

#include "Models.hpp"
#include "DataConverter.hpp"
#include "Utils.hpp"
#include <memory>
#include <torch/torch.h>

class Request;

class LocalBuffer {
public:
  LocalBuffer(torch::Tensor &_state, int numEnvs)
      : state(_state),
        transitions(numEnvs, Transition(_state, 1, SEQ_LENGTH, ACTION_SIZE)),
        indexes(numEnvs, 0),
        prevAction(numEnvs, torch::zeros({1}, torch::kUInt8)),
        prevReward(numEnvs, torch::zeros({1}, torch::kFloat32)),
        prevIh(numEnvs, torch::zeros({512}, torch::kFloat32)),
        prevHh(numEnvs, torch::zeros({512}, torch::kFloat32)) {}

  void setInferenceParam(int envId, Request &request, AgentInput *inferData);
  void updateAndGetTransition(int envId, Request &request,
                              torch::Tensor &action, AgentOutput & agentOutput,
                              torch::Tensor &policy,
                              std::vector<ReplayData> *retReplay,
                              std::vector<RetraceQ> *retRetrace);

  void getTransitions(std::vector<ReplayData> *retReplay,
                      std::vector<RetraceQ> *retRetrace);

private:
  torch::Tensor state;
  std::vector<Transition> transitions;
  std::vector<ReplayData> replayList;
  std::vector<RetraceQ> qList;
  std::vector<int> indexes;
  std::vector<torch::Tensor> prevAction;
  std::vector<torch::Tensor> prevReward;
  std::vector<torch::Tensor> prevIh;
  std::vector<torch::Tensor> prevHh;
};

#endif // LOCAL_BUFFER_HPP
