#include "LocalBuffer.hpp"
#include "Utils.hpp"

using namespace torch::indexing;
std::mutex mtx;

void LocalBuffer::setInferenceParam(Request &request, AgentInput *inferData) {

  inferData->state.index_put_(
      {0, 0},
      torch::from_blob(request.state, stateShape, torch::kUInt8) / 255.0);
  inferData->prevAction.index_put_({0}, prevAction);
  inferData->prevReward.index_put_({0}, prevReward);
}

void LocalBuffer::setRetaceData() {
  retraceData.action.index_put_(
      {retraceIndex}, torch::from_blob(transition.action + REPLAY_PERIOD,
                                       {1 + TRACE_LENGTH, 1}, torch::kUInt8)
                          .to(torch::kInt64)
                          .to(device));
  retraceData.reward.index_put_(
      {retraceIndex}, torch::from_blob(transition.reward + REPLAY_PERIOD,
                                       {1 + TRACE_LENGTH}, torch::kFloat32)
                          .to(device));
  retraceData.done.index_put_({retraceIndex},
                              torch::from_blob(transition.done + REPLAY_PERIOD,
                                               {1 + TRACE_LENGTH}, torch::kBool)
                                  .to(device));
  retraceData.policy.index_put_(
      {retraceIndex}, torch::from_blob(transition.policy + REPLAY_PERIOD,
                                       {1 + TRACE_LENGTH}, torch::kFloat32)
                          .to(device));
  retraceData.onlineQ.index_put_(
      {retraceIndex},
      torch::from_blob(transition.q + REPLAY_PERIOD,
                       {1 + TRACE_LENGTH, ACTION_SIZE}, torch::kFloat32)
          .to(device));
  retraceData.targetQ.index_put_(
      {retraceIndex},
      torch::from_blob(transition.q + REPLAY_PERIOD,
                       {1 + TRACE_LENGTH, ACTION_SIZE}, torch::kFloat32)
          .to(device));
  retraceIndex++;
}

bool LocalBuffer::updateAndGetTransition(Request &request,
                                         torch::Tensor &actionTensor,
                                         torch::Tensor &q,
                                         AgentOutput &agentOutput,
                                         torch::Tensor &policy) {
  auto action = actionTensor.item<u_int8_t>();

  prevAction = action;
  prevReward = request.reward;

  std::copy(request.state, request.state + STATE_SIZE, transition.state[index]);
  transition.action[index] = action;
  transition.reward[index] = request.reward;
  transition.done[index] = request.done;
  std::copy(prevIh, prevIh + LSTM_STATE_SIZE, transition.ih[index]);
  std::copy(prevHh, prevHh + LSTM_STATE_SIZE, transition.hh[index]);
  auto qBuf = q.contiguous().data_ptr<float>();
  std::copy(qBuf, qBuf + ACTION_SIZE, transition.q[index]);
  transition.policy[index] = policy.item<float>();

  // value <- batch(1), seq(1), value
  auto ihBuf = agentOutput.ih.cpu().contiguous().data_ptr<float>();
  std::copy(ihBuf, ihBuf + LSTM_STATE_SIZE, prevIh);
  auto hhBuf = agentOutput.hh.cpu().contiguous().data_ptr<float>();
  std::copy(hhBuf, hhBuf + LSTM_STATE_SIZE, prevHh);

  index++;

  if (index == SEQ_LENGTH || request.done) {
    // 報酬合計を取得
    auto totalReward =
        std::accumulate(transition.reward, transition.reward + index, 0.0);

    if (request.done) {
      std::fill(prevIh, prevIh + LSTM_STATE_SIZE, 0);
      std::fill(prevHh, prevHh + LSTM_STATE_SIZE, 0);
      prevAction = 0;
      prevReward = 0;

      if (index > REPLAY_PERIOD + 1) {
        // 現在位置から最後までゼロクリア
        std::fill(transition.state[index], transition.state[SEQ_LENGTH], 0);
        std::fill(&transition.action[index], &transition.action[SEQ_LENGTH], 0);
        std::fill(&transition.reward[index], &transition.reward[SEQ_LENGTH], 0);
        std::fill(&transition.done[index], &transition.done[SEQ_LENGTH], 0);
        std::fill(&transition.policy[index], &transition.policy[SEQ_LENGTH], 0);
        std::fill(transition.q[index], transition.q[SEQ_LENGTH], 0);

        // リトレースにデータ設定
        setRetaceData();

        // 遷移データを圧縮
        ReplayData &replayData = transition.getReplayData();
        storedDatas.emplace_back(std::move(compress(replayData)));
        storedDatas[retraceIndex].reward = totalReward;
      }
      // 現在位置をリセット
      index = 0;
    } else {
      // リトレースにデータ設定
      setRetaceData();

      // 遷移データを圧縮
      ReplayData &replayData = transition.getReplayData();
      storedDatas.emplace_back(std::move(compress(replayData)));
      storedDatas[retraceIndex].reward = totalReward;

      // 末尾からバーンイン期間分を先頭へ移動
      auto a = &transition.action[SEQ_LENGTH] - REPLAY_PERIOD;
      auto b = &transition.action;
      auto c = transition.state;

      std::move(transition.state[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.state[SEQ_LENGTH], (uint8_t *)transition.state);
      std::move(&transition.action[SEQ_LENGTH] - REPLAY_PERIOD,
                &transition.action[SEQ_LENGTH], (uint8_t *)transition.action);
      std::move(&transition.reward[SEQ_LENGTH] - REPLAY_PERIOD,
                &transition.reward[SEQ_LENGTH], (float *)transition.reward);
      std::move(&transition.done[SEQ_LENGTH] - REPLAY_PERIOD,
                &transition.done[SEQ_LENGTH], (bool *)transition.done);
      std::move(&transition.policy[SEQ_LENGTH] - REPLAY_PERIOD,
                &transition.policy[SEQ_LENGTH], (float *)transition.policy);
      std::move(transition.ih[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.ih[SEQ_LENGTH], (float *)transition.ih);
      std::move(transition.hh[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.hh[SEQ_LENGTH], (float *)transition.hh);
      std::move(transition.q[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.q[SEQ_LENGTH], (float *)transition.q);
      index = REPLAY_PERIOD;
    }
  }

  if (retraceIndex < BATCH_SIZE) {
    return false;
  } else {
    return true;
  }
}
