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
      : numEnvs(numEnvs_), actionSize(actionSize_),
        inferBatchSize(std::floor(numEnvs_ / 2)), agent(actionSize_),
        returnSize(returnSize_), state(state_),
        localBuffer(state_, numEnvs, returnSize_),
        dataConverter(state_, actionSize_, 1 + replayPeriod + traceLength),
        replay(dataConverter, capacity) {}

  int listenActor();
  int sendAndRecieveActor(int fd_other);
  int inference(int envId, Request &request, Event &event);
  Replay *getReplay() { return &replay; }
  void retraceLoop(Event &event);
  void trainLoop();

private:
  const int numEnvs;
  const int actionSize;
  const int inferBatchSize;
  const int returnSize;

  Agent agent;
  torch::Tensor state;
  std::vector<Request> requests;
  LocalBuffer localBuffer;
  DataConverter dataConverter;
  Replay replay;
};

#endif // LEARNER_HPP
