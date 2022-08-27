#include "LocalBuffer.hpp"

using namespace torch::indexing;
std::mutex mtx;

void LocalBuffer::setInferenceParam(int envId, Request &request,
                                    AgentInput *inferData) {

  inferData->state.index_put_({0, 0}, request.state / 255.0);
  inferData->prevAction.index_put_({0}, prevAction[envId].index({0}));
  inferData->prevReward.index_put_({0}, prevReward[envId].index({0}));
}

void LocalBuffer::updateAndGetTransition(int envId, Request &request,
                                         torch::Tensor &action,
                                         AgentOutput &agentOutput,
                                         torch::Tensor &policy,
                                         std::vector<ReplayData> *retReplay,
                                         std::vector<RetraceQ> *retRetrace) {

  prevAction[envId].index_put_({0}, action);
  prevReward[envId].index_put_({0}, request.reward);

  auto index = indexes[envId];

  transitions[envId].state.index_put_({0, index}, request.state);
  transitions[envId].action.index_put_({0, index}, action);
  transitions[envId].reward.index_put_({0, index}, request.reward);
  transitions[envId].done.index_put_({0, index}, request.done);
  transitions[envId].ih.index_put_({0, index}, prevIh[envId]);
  transitions[envId].hh.index_put_({0, index}, prevHh[envId]);
  transitions[envId].q.index_put_({0, index}, agentOutput.q.index({0, 0}));
  transitions[envId].policy.index_put_({0, index}, policy);

  // value <- batch(1), seq(1), value
  prevIh[envId].index_put_({Slice()}, agentOutput.ih.index({0, 0}));
  prevHh[envId].index_put_({Slice()}, agentOutput.hh.index({0, 0}));

  index++;

  if (index == SEQ_LENGTH || request.done) {
    if (request.done) {
      prevIh[envId].index_put_({Slice()}, 0);
      prevIh[envId].index_put_({Slice()}, 0);
      prevAction[envId].index_put_({0}, 0);
      prevReward[envId].index_put_({0}, 0);

      ReplayData replayData;
      RetraceQ retraceData;

      if (index > REPLAY_PERIOD + 1) {
        // 現在位置までコピー
        memcpy(replayData.state,
               transitions[envId]
                   .state.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<uint8_t>(),
               index * STATE_SIZE * sizeof(uint8_t));
        memcpy(replayData.action,
               transitions[envId]
                   .action.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<uint8_t>(),
               index * sizeof(uint8_t));
        memcpy(replayData.reward,
               transitions[envId]
                   .reward.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<float>(),
               index * sizeof(float));
        memcpy(replayData.done,
               transitions[envId]
                   .done.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<bool>(),
               index * sizeof(bool));
        memcpy(replayData.policy,
               transitions[envId]
                   .policy.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<float>(),
               index * sizeof(float));
        memcpy(retraceData.onlineQ,
               transitions[envId]
                   .q.index({0, Slice(None, index)})
                   .contiguous()
                   .data_ptr<float>(),
               index * ACTION_SIZE * sizeof(float));

        // lstm状態はシークなし
        memcpy(
            replayData.ih,
            transitions[envId].ih.index({0, 1}).contiguous().data_ptr<float>(),
            LSTM_STATE_SIZE * sizeof(float));
        memcpy(
            replayData.hh,
            transitions[envId].hh.index({0, 1}).contiguous().data_ptr<float>(),
            LSTM_STATE_SIZE * sizeof(float));

        // 現在地以降は0
        memset(replayData.state + index, 0,
               (SEQ_LENGTH - index) * STATE_SIZE * sizeof(uint8_t));
        memset(replayData.action + index, 0,
               (SEQ_LENGTH - index) * sizeof(uint8_t));
        memset(replayData.reward + index, 0,
               (SEQ_LENGTH - index) * sizeof(float));
        memset(replayData.done + index, 0, (SEQ_LENGTH - index) * sizeof(bool));
        memset(replayData.policy + index, 0,
               (SEQ_LENGTH - index) * sizeof(float));
        memset(retraceData.onlineQ + index, 0,
               (SEQ_LENGTH - index) * ACTION_SIZE * sizeof(float));
      }
      index = 0;

      std::lock_guard lock(mtx);
      replayList.emplace_back(std::move(replayData));
      qList.emplace_back(std::move(retraceData));
    } else {
      ReplayData replayData;
      RetraceQ retraceData;

      memcpy(replayData.state,
             transitions[envId].state.contiguous().data_ptr<uint8_t>(),
             SEQ_LENGTH * STATE_SIZE * sizeof(uint8_t));
      memcpy(replayData.action,
             transitions[envId].action.contiguous().data_ptr<uint8_t>(),
             SEQ_LENGTH * sizeof(uint8_t));
      memcpy(replayData.reward,
             transitions[envId].reward.contiguous().data_ptr<float>(),
             SEQ_LENGTH * sizeof(float));
      memcpy(replayData.done,
             transitions[envId].done.contiguous().data_ptr<bool>(),
             SEQ_LENGTH * sizeof(bool));
      memcpy(replayData.policy,
             transitions[envId].policy.contiguous().data_ptr<float>(),
             SEQ_LENGTH * sizeof(float));
      memcpy(retraceData.onlineQ,
             transitions[envId].q.contiguous().data_ptr<float>(),
             SEQ_LENGTH * ACTION_SIZE * sizeof(float));

      // lstm状態はシークなし
      memcpy(replayData.ih,
             transitions[envId].ih.index({0, 1}).contiguous().data_ptr<float>(),
             LSTM_STATE_SIZE * sizeof(float));
      memcpy(replayData.hh,
             transitions[envId].hh.index({0, 1}).contiguous().data_ptr<float>(),
             LSTM_STATE_SIZE * sizeof(float));

      index = 1 + REPLAY_PERIOD;

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

  std::lock_guard lock(mtx);
  if (replayList.size() >= BATCH_SIZE) {
    replayList.swap(*retReplay);
    qList.swap(*retRetrace);
  }
}
