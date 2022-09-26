#ifndef LEARNER_HPP
#define LEARNER_HPP

#include "Agent.hpp"
#include "LocalBuffer.hpp"
#include "Replay.hpp"

#include <vector>

const auto INVALID_ACTION = 99;

class Learner {
public:
  Learner(torch::Tensor state_, int actionSize_, int numEnvs_, int traceLength,
          int replayPeriod, int capacity)
      : numEnvs(numEnvs_), actionSize(actionSize_), state(state_),
        replay(capacity) {

    inferStateSizes = std::vector<int64_t>{1, 1};
    inferStateSizes.insert(inferStateSizes.end(), state_.sizes().begin(),
                           state_.sizes().end());

    for (int i = 0; i < NUM_TRAIN_THREADS; i++) {
      trainThread[i] = std::thread(&Learner::trainLoop, this, i);
    }
  }

  int listenActor();
  int sendAndRecieveActor(int fd_other, R2D2Agent inferModel);
  int inference(R2D2Agent &inferModel, Request &request, AgentInput &agentInput,
                torch::Device device, LocalBuffer &localBuffer);
  Replay *getReplay() { return &replay; }
  void trainLoop(int threadNum);

private:
  const int numEnvs;
  const int actionSize;

  int useIndex = 0;
  int nextUseIndex = 1;
  int freeIndex = 1;

  std::vector<int64_t> inferStateSizes;
  std::thread trainThread[NUM_TRAIN_THREADS];
  torch::Tensor state;
  Replay replay;
};

#endif // LEARNER_HPP
