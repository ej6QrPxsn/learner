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
    replayDataFuture = bufferNotification.get_future();

    addThread =
        std::thread(&Replay::addLoop, this, std::move(bufferNotification));
  }

  void updatePriorities(std::array<int, BATCH_SIZE> labels,
                        std::array<int, BATCH_SIZE> indexes,
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

  void putReplayQueue(torch::Tensor priorities, std::vector<StoredData> data) {
    if (replayQueue.size() < MAX_REPLAY_QUEUE_SIZE) {
      replayQueue.emplace_back(
          std::move(std::make_tuple(priorities, std::move(data))));
      replayCond.notify_one();
    }
  }

  float median(std::vector<float> &v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
  }

  auto popReplayQueue() {
    std::unique_lock<std::mutex> lck(replayMtx);
    replayCond.wait(lck, [&] { return !replayQueue.empty(); });
    auto queueData = std::move(replayQueue.front());
    replayQueue.pop_front();
    return queueData;
  }

  void addReplay() {
    auto queueData = std::move(popReplayQueue());

    auto priorities = std::get<0>(queueData);
    auto dataList = std::move(std::get<1>(queueData));

    for (int i = 0; i < dataList.size(); i++) {
      auto &storeData = dataList[i];
      auto reward = storeData.reward;
      auto size = storeData.size;
      auto ptr = storeData.ptr.get();
      replayBuffer.add(priorities.index({i}).item<float>(),
                       std::move(storeData));

      // 遷移の報酬が高報酬リストの中央値よりも高いなら、高報酬バッファに遷移を入れる
      const auto medVal = median(highRewards);
      if (reward > medVal) {
        StoredData data;
        data.size = size;
        data.reward = reward;
        data.ptr = std::make_unique<char[]>(size);
        memcpy(data.ptr.get(), ptr, size);
        highRewardBuffer.add(priorities.index({i}).item<float>(),
                             std::move(data));

        // 高報酬リストの最小値を新しい報酬で置き換える
        auto minIter = min_element(highRewards.begin(), highRewards.end());
        *minIter = reward;
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
      for (int i = replayCount; i < BATCH_SIZE; i++) {
        sampleData.labelList[i] = HIGH_REWARD;
      }
    }
  }

  void sample(SampleData &sampleData) {
    replayDataFuture.wait();

    getSample(sampleData);
  }

  std::random_device rnd;
  std::mt19937 engine;
  std::uniform_real_distribution<> dist;

  std::future<void> replayDataFuture;
  std::thread addThread;
  std::thread sampleThread;
  DataConverter dataConverter;
  ReplayBuffer replayBuffer;
  ReplayBuffer highRewardBuffer;
  std::vector<float> highRewards;

  std::deque<std::tuple<torch::Tensor, std::vector<StoredData>>> replayQueue;
  std::mutex replayMtx;
  std::condition_variable replayCond;
  std::mutex sampleMtx;
  std::condition_variable sampleCond;
  bool isSample = false;
};

#endif // REPLAY_HPP
