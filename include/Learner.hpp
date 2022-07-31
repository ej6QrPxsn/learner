#ifndef LEARNER_HPP
#define LEARNER_HPP

#include "Agent.hpp"
#include "DataConverter.hpp"
#include "LocalBuffer.hpp"
#include "Replay.hpp"
#include "Request.hpp"
#include <vector>

const auto INVALID_ACTION = 99;

class Learner {
public:
  Learner(torch::Tensor state_, int actionSize_, int numEnvs_, int traceLength,
          int replayPeriod, int returnSize_, int capacity)
      : numEnvs(numEnvs_), actionSize(actionSize_), agent(actionSize_),
        returnSize(returnSize_), state(state_),
        localBuffer(state_, numEnvs, returnSize_),
        dataConverter(state_, actionSize_, 1 + replayPeriod + traceLength),
        replay(dataConverter, capacity) {}

  int listenActor();
  int sendAndRecieveActor(int fd_other);
  int inference(int envId, Request &request);
  Replay *getReplay() { return &replay; }
  void trainLoop();

private:
  const int numEnvs;
  const int actionSize;
  const int returnSize;

  Agent agent;
  torch::Tensor state;
  LocalBuffer localBuffer;
  DataConverter dataConverter;
  Replay replay;
};

#endif // LEARNER_HPP
