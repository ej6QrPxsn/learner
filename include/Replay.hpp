#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "ReplayBuffer.hpp"
#include <deque>
#include <future>
#include <mutex>
#include <numeric>
#include <random>

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

  void updatePriorities(std::array<int, BATCH_SIZE> labels, std::array<int, BATCH_SIZE> indexes,
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
      replayBuffer.add(priorities.index({i}).item<float>(), dataList[i]);

      // 遷移の報酬が高報酬リストの平均よりも高いなら、高報酬バッファに遷移を入れる
      auto rewards = std::accumulate(dataList[i].reward,
                                     dataList[i].reward + SEQ_LENGTH, 0.0);
      const auto ave =
          std::accumulate(std::begin(highRewards), std::end(highRewards), 0.0) /
          std::size(highRewards);
      if (rewards > ave) {
        highRewardBuffer.add(priorities.index({i}).item<float>(), dataList[i]);

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

  void getSample(SampleData &sampleData) {
    int highRewardCount = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
      auto rand = dist(engine);
      if (rand < HIGH_REWARD_RATIO) {
        highRewardCount++;
      }
    }
    auto replayCount = BATCH_SIZE - highRewardCount;

    if (replayCount > 0) {
      replayBuffer.sample(replayCount, sampleData, 0);
      sampleData.labelList.fill(REPLAY);
    }

    if (highRewardCount > 0) {
      highRewardBuffer.sample(highRewardCount, sampleData, replayCount);
      for(int i = replayCount; i < BATCH_SIZE; i++) {
        sampleData.labelList[i] = HIGH_REWARD;
      }
    }

  }

  void sample(SampleData &sampleData) {
    ReplayDataFuture.wait();

    getSample(sampleData);
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
};

#endif // REPLAY_HPP
