#include "DataConverter.hpp"
#include "Common.hpp"

TrainData DataConverter::toBatchedTrainData(std::vector<ReplayData> &dataList) {
  int batchSize = dataList.size();
  TrainData train(state, batchSize, seqLength);

  for (int i = 0; i < batchSize; i++) {
    train.state.index_put_({i}, dataList[i].state.index({0}) / 255.0);
    train.action.index_put_({i}, dataList[i].action.index({0, Slice(), 0}));
    train.reward.index_put_({i}, dataList[i].reward.index({0}));
    train.done.index_put_({i}, dataList[i].done.index({0}));
    train.ih.index_put_({i}, dataList[i].ih.index({0}));
    train.hh.index_put_({i}, dataList[i].hh.index({0}));
    train.policy.index_put_({i}, dataList[i].policy.index({0, Slice(), 0}));
  }

  train.ih.requires_grad();
  train.hh.requires_grad();
  return std::move(train);
}

RetraceData DataConverter::toBatchedRetraceData(std::vector<ReplayData> &replayDatas, std::vector<RetraceQ> &RetraceQs) {
  int batchSize = replayDatas.size();
  RetraceData retrace(batchSize, 1 + TRACE_LENGTH, actionSize);

  for (int i = 0; i < batchSize; i++) {
    retrace.action.index_put_({i}, replayDatas[i].action.index({0, Slice(REPLAY_PERIOD, None)}));
    retrace.reward.index_put_({i}, replayDatas[i].reward.index({0, Slice(REPLAY_PERIOD, None), 0}));
    retrace.done.index_put_({i}, replayDatas[i].done.index({0, Slice(REPLAY_PERIOD, None), 0}));
    retrace.policy.index_put_({i}, replayDatas[i].policy.index({0, Slice(REPLAY_PERIOD, None), 0}));
    retrace.onlineQ.index_put_({i}, RetraceQs[i].onlineQ.index({0, Slice(REPLAY_PERIOD, None)}));
    retrace.targetQ.index_put_({i}, RetraceQs[i].onlineQ.index({0, Slice(REPLAY_PERIOD, None)}));
  }
  return retrace;
}

