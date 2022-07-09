#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "ReplayBuffer.hpp"
#include "Utils.hpp"
#include "Request.hpp"
#include <deque>
#include <memory>
#include <mutex>

class Replay {
public:
  Replay(DataConverter converter, int capacity)
      : dataConverter(converter), buffer(ReplayBuffer(capacity)) {}

  inline void putReplayQueue(torch::Tensor priorities,
                             std::vector<ReplayData> data) {
    std::lock_guard<std::mutex> lock(replayMtx);
    replayQueue.push_back(std::make_tuple(priorities, data));
  }

  inline void putPriorityQueue(std::vector<int> indexes,
                               torch::Tensor priorities) {
    std::lock_guard<std::mutex> lock(priorityMtx);
    priorityQueue.push_back(std::make_tuple(indexes, priorities));
  }

  inline std::tuple<std::vector<int>, TrainData> sample() {
    event.wait();
    std::lock_guard<std::mutex> lock(sampleMtx);
    auto sample = std::move(sampleQueue.front());
    sampleQueue.pop_front();
    return sample;
  }

  void loop() {

    while (1) {
      {
        if (!replayQueue.empty()) {
          std::tuple<torch::Tensor, std::vector<ReplayData>> queueData;
          {
            std::lock_guard<std::mutex> lock(replayMtx);
            queueData = replayQueue.front();
            replayQueue.pop_front();
          }

          auto priorities = std::get<0>(queueData);
          auto dataList = std::get<1>(queueData);

          for (int i = 0; i < dataList.size(); i++) {
            buffer.add(priorities.index({i}).item<float>(), std::move(dataList[i]));
          }
        }

        if (buffer.get_count() > REPLAY_BUFFER_MIN_SIZE && sampleQueue.empty()) {
          auto sample = buffer.sample(BATCH_SIZE);
          auto indexes = std::get<0>(sample);
          auto data = dataConverter.toBatchedTrainData(std::get<1>(sample));

          {
            std::lock_guard<std::mutex> lock(sampleMtx);
            sampleQueue.push_back(std::move(std::make_tuple(indexes, data)));
          }

          if (!event.notify) {
            event.set();
          }
        }

        if (!priorityQueue.empty()) {
          std::tuple<std::vector<int>, torch::Tensor> queueData;
          {
            std::lock_guard<std::mutex> lock(priorityMtx);
            queueData = priorityQueue.front();
            priorityQueue.pop_front();
          }

          auto indexes = std::get<0>(queueData);
          auto priorities = std::get<1>(queueData);

          for (int i = 0; i < indexes.size(); i++) {
            buffer.update(indexes[i], priorities.index({i}).item<float>());
          }
        }
      }
    }
  }

  DataConverter dataConverter;
  ReplayBuffer buffer;
  std::mutex replayMtx;
  std::mutex sampleMtx;
  std::mutex priorityMtx;
  Event event;
  std::deque<std::tuple<torch::Tensor, std::vector<ReplayData>>> replayQueue;
  std::deque<std::tuple<std::vector<int>, TrainData>> sampleQueue;
  std::deque<std::tuple<std::vector<int>, torch::Tensor>> priorityQueue;
};

#endif // REPLAY_HPP
