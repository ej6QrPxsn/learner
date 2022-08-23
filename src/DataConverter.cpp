#include "DataConverter.hpp"
#include "Common.hpp"

TrainData DataConverter::toBatchedTrainData(std::vector<ReplayData> dataList) {
  int batchSize = dataList.size();
  TrainData train(state, batchSize, seqLength);

  for (int i = 0; i < batchSize; i++) {
    train.state.index_put_({i}, torch::from_blob(dataList[i].state,
                                                 train.state.index({i}).sizes(),
                                                 torch::kUInt8) /
                                    255.0);
    train.action.index_put_(
        {i}, torch::from_blob(dataList[i].action,
                              train.action.index({i}).sizes(), torch::kUInt8));
    train.reward.index_put_(
        {i}, torch::from_blob(dataList[i].reward,
                              train.reward.index({i}).sizes(), torch::kFloat));
    train.done.index_put_({i}, torch::from_blob(dataList[i].done,
                                                train.done.index({i}).sizes(),
                                                torch::kBool));
    train.ih.index_put_({i}, torch::from_blob(dataList[i].ih,
                                              train.ih.index({i}).sizes(),
                                              torch::kFloat));
    train.hh.index_put_({i}, torch::from_blob(dataList[i].hh,
                                              train.hh.index({i}).sizes(),
                                              torch::kFloat));
    train.policy.index_put_(
        {i}, torch::from_blob(dataList[i].policy,
                              train.policy.index({i}).sizes(), torch::kFloat));
  }

  train.ih.requires_grad();
  train.hh.requires_grad();
  return std::move(train);
}

void DataConverter::toBatchedRetraceData(std::vector<ReplayData> &replayDatas,
                                         std::vector<RetraceQ> &RetraceQs,
                                         RetraceData *retrace, int batchSize) {
  for (int i = 0; i < batchSize; i++) {
    retrace->action.index_put_(
        {i}, torch::from_blob(replayDatas[i].action, {SEQ_LENGTH, 1}, torch::kUInt8)
                 .index({Slice(REPLAY_PERIOD, None)}));
    retrace->reward.index_put_(
        {i}, torch::from_blob(replayDatas[i].reward, SEQ_LENGTH, torch::kFloat)
                 .index({Slice(REPLAY_PERIOD, None)}));
    retrace->done.index_put_(
        {i}, torch::from_blob(replayDatas[i].done, SEQ_LENGTH, torch::kBool)
                 .index({Slice(REPLAY_PERIOD, None)}));
    retrace->policy.index_put_(
        {i}, torch::from_blob(replayDatas[i].policy, SEQ_LENGTH, torch::kFloat)
                 .index({Slice(REPLAY_PERIOD, None)}));
    retrace->onlineQ.index_put_({i}, torch::from_blob(RetraceQs[i].onlineQ,
                                                      {SEQ_LENGTH, ACTION_SIZE},
                                                      torch::kFloat)
                                         .index({Slice(REPLAY_PERIOD, None)}));
    retrace->targetQ.index_put_({i}, torch::from_blob(RetraceQs[i].onlineQ,
                                                      {SEQ_LENGTH, ACTION_SIZE},
                                                      torch::kFloat)
                                         .index({Slice(REPLAY_PERIOD, None)}));
  }
}
