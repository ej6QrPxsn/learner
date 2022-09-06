#include "DataConverter.hpp"

void DataConverter::toBatchedTrainData(
    TrainData &train, std::array<ReplayData, BATCH_SIZE> &dataList) {

  for (int i = 0; i < BATCH_SIZE; i++) {
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
}
