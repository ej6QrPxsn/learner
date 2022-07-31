#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "ReplayBuffer.hpp"
#include <deque>
#include <future>
#include <mutex>
#include <random>
#include <numeric>


class Replay {
public:
  Replay(DataConverter converter, int capacity)
      : dataConverter(converter), replayBuffer(capacity), engine(rnd()),
        dist(0.0, 1.0), highRewards(HIGH_REWARD_SIZE, 0),
        highRewardBuffer(HIGH_REWARD_BUFFER_SIZE) {
    std::promise<void> bufferNotification;
    ReplayDataFuture = bufferNotification.get_future();

    addThread =
        std::thread(&Replay::addLoop, this, std::move(bufferNotification));
  }

  void updatePriorities(std::vector<int> labels, std::vector<int> indexes,
                        torch::Tensor priorities) {
    for (int i = 0; i < indexes.size(); i++) {
      if (labels[i] == REPLAY) {
        replayBuffer.update(indexes[i], priorities.index({i}).item<float>());
      } else {
        highRewardBuffer.update(indexes[i],
                                priorities.index({i}).item<float>());
      }
    }
  }

  void putReplayQueue(torch::Tensor priorities, std::vector<ReplayData> data) {
    if (replayQueue.size() < MAX_REPLAY_QUEUE_SIZE) {
      replayQueue.emplace_back(std::move(std::make_tuple(priorities, data)));
      replayCond.notify_one();
    }
  }

  auto popReplayQueue() {
    std::unique_lock<std::mutex> lck(replayMtx);
    replayCond.wait(lck, [&] { return !replayQueue.empty(); });
    auto queueData = replayQueue.front();
    replayQueue.pop_front();
    return queueData;
  }

  void addReplay() {
    auto queueData = std::move(popReplayQueue());

    auto priorities = std::get<0>(queueData);
    auto dataList = std::get<1>(queueData);

    for (int i = 0; i < dataList.size(); i++) {
      replayBuffer.add(priorities.index({i}).item<float>(), dataList[i].clone());

      // 遷移の報酬が高報酬リストの平均よりも高いなら、高報酬バッファに遷移を入れる
      auto rewards = torch::sum(dataList[i].reward).item<float>();
      const auto ave = std::accumulate(std::begin(highRewards), std::end(highRewards), 0.0) / std::size(highRewards);
      if (rewards > ave) {
        highRewardBuffer.add(priorities.index({i}).item<float>(), dataList[i].clone());

        // 高報酬リストの最小値を新しい報酬で置き換える
        auto minIter = min_element(highRewards.begin(), highRewards.end());
        *minIter = rewards;
      }
    }
  }

  void addLoop(std::promise<void> ReplayDataPromise) {
    while (replayBuffer.get_count() < REPLAY_BUFFER_MIN_SIZE) {
      addReplay();
    }

    ReplayDataPromise.set_value();

    while (1) {
      addReplay();
    }
  }

  auto getSample() {
    int highRewardCount = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
      auto rand = dist(engine);
      if (rand < HIGH_REWARD_RATIO) {
        highRewardCount++;
      }
    }
    auto replay_count = BATCH_SIZE - highRewardCount;

    std::vector<ReplayData> dataList;
    std::vector<int> indexList;
    std::vector<int> labelList;

    if (highRewardCount > 0) {
      auto sample = replayBuffer.sample(highRewardCount);
      auto indexes = std::get<0>(sample);
      indexList.insert(indexList.end(), indexes.begin(), indexes.end());
      auto data = std::get<1>(sample);
      dataList.insert(dataList.end(), data.begin(), data.end());
      auto label = std::vector<int>(highRewardCount, HIGH_REWARD);
      labelList.insert(labelList.end(), label.begin(), label.end());
    }

    if (replay_count > 0) {
      auto sample = replayBuffer.sample(replay_count);
      auto indexes = std::get<0>(sample);
      indexList.insert(indexList.end(), indexes.begin(), indexes.end());
      auto data = std::get<1>(sample);
      dataList.insert(dataList.end(), data.begin(), data.end());
      auto label = std::vector<int>(replay_count, REPLAY);
      labelList.insert(labelList.end(), label.begin(), label.end());
    }

    auto data = dataConverter.toBatchedTrainData(dataList);
    return std::make_tuple(labelList, indexList, data);
  }

  auto sample() {
    ReplayDataFuture.wait();

    return getSample();
  }

  std::random_device rnd;
  std::mt19937 engine;
  std::uniform_real_distribution<> dist;

  std::future<void> ReplayDataFuture;
  std::thread addThread;
  std::thread sampleThread;
  DataConverter dataConverter;
  ReplayBuffer replayBuffer;
  ReplayBuffer highRewardBuffer;
  std::vector<float> highRewards;

  std::deque<std::tuple<torch::Tensor, std::vector<ReplayData>>> replayQueue;
  std::mutex replayMtx;
  std::condition_variable replayCond;
  std::mutex sampleMtx;
  std::condition_variable sampleCond;
  bool isSample = false;
  std::tuple<std::vector<int>, std::vector<int>, TrainData> sampleData;
};

#endif // REPLAY_HPP
