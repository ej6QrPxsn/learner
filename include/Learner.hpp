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
  Learner(torch::Tensor state_, int actionSize, int numEnvs_, int traceLength,
          int replayPeriod, int returnSize, int capacity)
      : numEnvs(numEnvs_), inferBatchSize(std::floor(numEnvs_ / 2)),
        agent(actionSize), state(state_),
        reqManager(numEnvs, inferBatchSize, state_),
        localBuffer(state_, actionSize, numEnvs, traceLength, replayPeriod,
                    returnSize),
        dataConverter(state_, actionSize, 1 + replayPeriod + traceLength),
        replay(dataConverter, capacity), actions(numEnvs, INVALID_ACTION) {}

  int listenActor();
  int sendAndRecieveActor(int fd_other);
  void inference(std::vector<int> &envIds);
  Replay *getReplay() { return &replay; }
  void trainLoop();

private:
  int numEnvs;
  int inferBatchSize;
  Agent agent;
  torch::Tensor state;
  RequestManager reqManager;
  std::vector<Request> requests;
  LocalBuffer localBuffer;
  DataConverter dataConverter;
  Replay replay;
  std::vector<int> actions;
};

#endif // LEARNER_HPP
