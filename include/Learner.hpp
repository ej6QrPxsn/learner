#ifndef LEARNER_HPP
#define LEARNER_HPP

#include "Agent.hpp"
#include "DataConverter.hpp"
#include "LocalBuffer.hpp"
#include "Replay.hpp"

#include <vector>

const auto INVALID_ACTION = 99;

class Learner {
public:
  Learner(torch::Tensor state_, int actionSize_, int numEnvs_, int traceLength,
          int replayPeriod, int capacity)
      : numEnvs(numEnvs_), actionSize(actionSize_), agent(actionSize_),
        state(state_),
        dataConverter(state_, actionSize_, 1 + replayPeriod + traceLength),
        replay(dataConverter, capacity) {}

  int listenActor();
  int sendAndRecieveActor(int fd_other);
  int inference(Request &request, AgentInput &agentInput,
                torch::Device device, LocalBuffer &localBuffer);
  Replay *getReplay() { return &replay; }
  void trainLoop();

private:
  const int numEnvs;
  const int actionSize;

  int useIndex = 0;
  int nextUseIndex = 1;
  int freeIndex = 1;

  Agent agent;
  torch::Tensor state;
  DataConverter dataConverter;
  Replay replay;
};

#endif // LEARNER_HPP
