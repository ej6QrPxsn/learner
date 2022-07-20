#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "ReplayBuffer.hpp"
#include <deque>
#include <future>
#include <mutex>

class Replay {
public:
  Replay(DataConverter converter, int capacity)
      : dataConverter(converter), buffer(ReplayBuffer(capacity)) {
    std::promise<void> bufferNotification;
    waitForNotification = bufferNotification.get_future();

    addThread =
        std::thread(&Replay::addLoop, this, std::move(bufferNotification));
    sampleThread =
        std::thread(&Replay::sampleLoop, this);
  }

  void updatePriorities(std::vector<int> indexes, torch::Tensor priorities) {
    for (int i = 0; i < indexes.size(); i++) {
      buffer.update(indexes[i], priorities.index({i}).item<float>());
    }
  }

  void putReplayQueue(torch::Tensor priorities, std::vector<ReplayData> data) {
    if (replayQueue.size() < MAX_REPLAY_QUEUE_SIZE) {
      replayQueue.emplace_back(std::move(std::make_tuple(priorities, data)));
      replayCond.notify_one();
    }
  }

  void addReplay() {
    {
      std::unique_lock<std::mutex> lck(replayMtx);
      replayCond.wait(lck, [&] { return !replayQueue.empty(); });
    }

    auto queueData = std::move(replayQueue.front());
    replayQueue.pop_front();

    auto priorities = std::get<0>(queueData);
    auto dataList = std::get<1>(queueData);

    for (int i = 0; i < dataList.size(); i++) {
      buffer.add(priorities.index({i}).item<float>(), dataList[i]);
    }
  }

  void addLoop(std::promise<void> bufferNotification) {
    while (buffer.get_count() < REPLAY_BUFFER_MIN_SIZE) {
      addReplay();
    }

    getSample();

    bufferNotification.set_value();

    while (1) {
      addReplay();
    }
  }

  void getSample() {
      auto sample = buffer.sample(BATCH_SIZE);
      auto indexes = std::get<0>(sample);
      auto data = dataConverter.toBatchedTrainData(std::get<1>(sample));
      sampleData = std::make_tuple(indexes, data);
  }

  void sampleLoop() {
    while (1) {
      {
        std::unique_lock<std::mutex> lck(sampleMtx);
        sampleCond.wait(lck, [&] { return isSample; });
      }
      getSample();
      isSample = false;
    }
  }

  auto sample() {
    waitForNotification.wait();

    auto data = std::move(sampleData);

    isSample = true;
    sampleCond.notify_one(); 

    return data;
  }

  std::future<void> waitForNotification;
  std::thread addThread;
  std::thread sampleThread;
  DataConverter dataConverter;
  ReplayBuffer buffer;
  std::deque<std::tuple<torch::Tensor, std::vector<ReplayData>>> replayQueue;
  std::mutex replayMtx;
  std::condition_variable replayCond;
  std::mutex sampleMtx;
  std::condition_variable sampleCond;
  bool isSample = false;
  std::tuple<std::vector<int>, TrainData> sampleData;
};

#endif // REPLAY_HPP
