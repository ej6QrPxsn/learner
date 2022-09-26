#include "LocalBuffer.hpp"
#include "Utils.hpp"

using namespace torch::indexing;

void LocalBuffer::setInferenceParam(Request &request, AgentInput *inferData) {

  inferData->state.index_put_(
      {0, 0},
      torch::from_blob(request.state, stateShape, torch::kUInt8) / 255.0);

  inferData->prevAction.index_put_({0}, prevAction);
  inferData->prevReward.index_put_({0}, prevReward);

  // ここから以前へ遡る必要はないので、念のため、detachしておく
  inferData->hiddenStates = prevHiddenStates.detach();
  inferData->cellStates = prevCellStates.detach();
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
                                         LstmStates &lstmStates,
                                         torch::Tensor &policy) {
  auto action = actionTensor.item<int>();
  prevAction = action;
  prevReward = request.reward;

  std::copy(request.state, request.state + STATE_SIZE, transition.state[index]);
  transition.action[index] = action;
  transition.reward[index] = request.reward;
  transition.done[index] = request.done;

  // ここで受け取るLSTM状態は、推論後の最新のものなので、一つ前の状態を設定する
  auto prevHiddenStatesBuf = prevHiddenStates.contiguous().data_ptr<float>();
  std::copy(prevHiddenStatesBuf, prevHiddenStatesBuf + LSTM_STATE_SIZE,
            transition.hiddenStates[index]);
  auto prevCellStatesBuf = prevCellStates.contiguous().data_ptr<float>();
  std::copy(prevCellStatesBuf, prevCellStatesBuf + LSTM_STATE_SIZE,
            transition.cellStates[index]);

  auto qBuf = q.contiguous().data_ptr<float>();
  std::copy(qBuf, qBuf + ACTION_SIZE, transition.q[index]);
  transition.policy[index] = policy.item<float>();

  auto [hiddenStates, cellStates] = lstmStates;
  prevHiddenStates = hiddenStates.detach().clone();
  prevCellStates = cellStates.detach().clone();

  index++;

  if (index == SEQ_LENGTH || request.done) {
    // 報酬合計を取得
    auto totalReward =
        std::accumulate(transition.reward, transition.reward + index, 0.0);

    if (request.done) {
      prevHiddenStates = torch::zeros({1, LSTM_STATE_SIZE});
      prevCellStates = torch::zeros({1, LSTM_STATE_SIZE});

      prevAction = 0;
      prevReward = 0;

      if (index > REPLAY_PERIOD + 1) {
        // 現在位置（次の現在位置）から最後までゼロクリア
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
      std::move(transition.hiddenStates[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.hiddenStates[SEQ_LENGTH],
                (float *)transition.hiddenStates);
      std::move(transition.cellStates[SEQ_LENGTH] - REPLAY_PERIOD,
                transition.cellStates[SEQ_LENGTH],
                (float *)transition.cellStates);
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
