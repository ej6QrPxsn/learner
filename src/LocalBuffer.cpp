#include "LocalBuffer.hpp"
#include "Learner.hpp"

using namespace torch::indexing;
std::mutex mtx;

void LocalBuffer::setInferenceParam(std::vector<int> &envIds,
                                    InferenceData *inferData,
                                    std::vector<Request> &requests) {
  int64_t size = envIds.size();

  for (int i = 0; i < size; i++) {
    auto envId = envIds[i];
    inferData->state.index_put_({i, 0}, requests[envId].state / 255.0);
    auto index = indexes[envId];
    if (index > 0) {
      inferData->prevAction.index_put_(
          {i}, transitions[envId].action.index({0, index - 1}));
      inferData->PrevReward.index_put_(
          {i}, transitions[envId].reward.index({0, index - 1}));
    } else {
      inferData->prevAction.index_put_({i}, 0);
      inferData->PrevReward.index_put_({i}, 0);
    }
  }
}

void LocalBuffer::updateAndGetTransitions(
    std::vector<int> &envIds, std::vector<Request> &requests,
    torch::Tensor &action, std::tuple<torch::Tensor, torch::Tensor> &lstmStates,
    torch::Tensor &q, torch::Tensor &policy, std::vector<ReplayData> *retReplay,
    std::vector<RetraceQ> *retRetrace) {
  auto ih = std::get<0>(lstmStates).permute({1, 0, 2}).clone();
  auto hh = std::get<1>(lstmStates).permute({1, 0, 2}).clone();

  for (int i = 0; i < envIds.size(); i++) {
    auto envId = envIds[i];
    auto index = indexes[envId];
    transitions[envId].state.index_put_({0, index}, requests[envId].state);
    transitions[envId].action.index_put_({0, index}, action.index({i}));
    transitions[envId].reward.index_put_({0, index}, requests[envId].reward);
    transitions[envId].done.index_put_({0, index}, requests[envId].done);
    transitions[envId].ih.index_put_({0, index}, prevIh[envId]);
    transitions[envId].hh.index_put_({0, index}, prevHh[envId]);
    transitions[envId].q.index_put_({0, index}, q.index({i}));
    transitions[envId].policy.index_put_({0, index}, policy.index({i}));

    prevIh[envId] = ih.index({i});
    prevHh[envId] = hh.index({i});

    index++;

    if (index == seqLength || requests[envId].done) {
      if (requests[envId].done) {
        prevIh[envId].index_put_({Slice()}, 0);
        prevIh[envId].index_put_({Slice()}, 0);

        ReplayData replayData = ReplayData(state, 1, seqLength);
        RetraceQ retraceData = RetraceQ(1, seqLength, actionSize);

        if (index > replayPeriod + 1) {
          // 現在位置までコピー
          replayData.state.index_put_(
              {0, Slice(None, index)},
              transitions[envId].state.index({0, Slice(None, index)}));
          replayData.action.index_put_(
              {0, Slice(None, index)},
              transitions[envId].action.index({0, Slice(None, index)}));
          replayData.reward.index_put_(
              {0, Slice(None, index)},
              transitions[envId].reward.index({0, Slice(None, index)}));
          replayData.done.index_put_(
              {0, Slice(None, index)},
              transitions[envId].done.index({0, Slice(None, index)}));
          replayData.policy.index_put_(
              {0, Slice(None, index)},
              transitions[envId].policy.index({0, Slice(None, index)}));
          retraceData.onlineQ.index_put_(
              {0, Slice(None, index)},
              transitions[envId].q.index({0, Slice(None, index)}));

          // lstm状態はシークなし
          replayData.ih.index_put_({0}, transitions[envId].ih.index({0, 1}));
          replayData.hh.index_put_({0}, transitions[envId].hh.index({0, 1}));

          // 現在地以降は0
          replayData.state.index_put_({0, Slice(index, None)}, 0);
          replayData.action.index_put_({0, Slice(index, None)}, 0);
          replayData.reward.index_put_({0, Slice(index, None)}, 0);
          replayData.done.index_put_({0, Slice(index, None)}, 0);
          replayData.policy.index_put_({0, Slice(index, None)}, 0);
          retraceData.onlineQ.index_put_({0, Slice(index, None)}, 0);
        }
        index = 0;

        std::lock_guard lock(mtx);
        replayList.emplace_back(std::move(replayData));
        qList.emplace_back(std::move(retraceData));
      } else {
        ReplayData replayData = ReplayData(state, 1, seqLength);
        RetraceQ retraceData = RetraceQ(1, seqLength, actionSize);

        replayData.state = transitions[envId].state.clone();
        replayData.action = transitions[envId].action.clone();
        replayData.reward = transitions[envId].reward.clone();
        replayData.done = transitions[envId].done.clone();
        replayData.policy = transitions[envId].policy.clone();
        retraceData.onlineQ = transitions[envId].q.clone();

        // lstm状態はシークなし
        replayData.ih.index_put_({0}, transitions[envId].ih.index({0, 1}));
        replayData.hh.index_put_({0}, transitions[envId].hh.index({0, 1}));

        index = 1 + replayPeriod;

        // 末尾からバーンイン期間分を先頭へ移動
        transitions[envId].state.index_put_(
            {0, Slice(None, index)},
            transitions[envId].state.index({0, Slice(-index, None)}));
        transitions[envId].action.index_put_(
            {0, Slice(None, index)},
            transitions[envId].action.index({0, Slice(-index, None)}));
        transitions[envId].reward.index_put_(
            {0, Slice(None, index)},
            transitions[envId].reward.index({0, Slice(-index, None)}));
        transitions[envId].done.index_put_(
            {0, Slice(None, index)},
            transitions[envId].done.index({0, Slice(-index, None)}));
        transitions[envId].ih.index_put_(
            {0, Slice(None, index)},
            transitions[envId].ih.index({0, Slice(-index, None)}));
        transitions[envId].hh.index_put_(
            {0, Slice(None, index)},
            transitions[envId].hh.index({0, Slice(-index, None)}));
        transitions[envId].q.index_put_(
            {0, Slice(None, index)},
            transitions[envId].q.index({0, Slice(-index, None)}));
        transitions[envId].policy.index_put_(
            {0, Slice(None, index)},
            transitions[envId].policy.index({0, Slice(-index, None)}));

        std::lock_guard lock(mtx);
        replayList.emplace_back(std::move(replayData));
        qList.emplace_back(std::move(retraceData));
      }
    }

    indexes[envId] = index;
  }

  std::lock_guard lock(mtx);
  if (replayList.size() > returnSize) {
    replayList.swap(*retReplay);
    qList.swap(*retRetrace);
  }
}
