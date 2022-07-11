#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "Request.hpp"
#include "SumTree.hpp"
#include "Utils.hpp"
#include <deque>
#include <memory>
#include <mutex>
#include <random>

struct SampleData {
  std::vector<int> indexes;
  TrainData datas;

  SampleData(std::vector<int> i, TrainData d) : indexes(i), datas(d) {}
};

class ReplayBuffer {
public:
  ReplayBuffer(int capacity) : tree(SumTree(capacity)), count(0) {}

  int get_count() { return count; }

  void update(int idx, float p) { tree.update(idx, p); }

  void add(float p, ReplayData sample) {
    assert(sample.action.size(0) != 0);
    tree.add(p, sample);
    count += 1;
    if (count < REPLAY_BUFFER_MIN_SIZE) {
      if (count % REPLAY_BUFFER_ADD_PRINT_SIZE == 0) {
        std::cout << "Waiting for the replay buffer to fill up. "
                  << "It currently has " << count;
        std::cout << " elements, waiting for at least "
                  << REPLAY_BUFFER_MIN_SIZE << " elements" << std::endl;
      }
    }
  }

  std::tuple<std::vector<int>, std::vector<ReplayData>> sample(int n) {
    std::vector<int> idx_list;
    std::vector<ReplayData> data_list;

    auto segment = tree.total() / n;

    for (auto i = 0; i < n; i++) {
      auto a = segment * i;
      auto b = segment * (i + 1);

      std::random_device rd;
      std::default_random_engine eng(rd());
      std::uniform_int_distribution<int> distr(a, b);

      auto s = distr(eng);
      if (s == 0) {
        s = 1;
      }
      auto ret = tree.get(s);
      auto index = std::get<0>(ret);
      auto data = std::get<2>(ret);

      assert(data.action.size(0) != 0);

      //(idx, p, data)
      idx_list.push_back(index);
      data_list.push_back(data);
    }

    return {idx_list, data_list};
  }

private:
  SumTree tree;
  int count;
};

class Replay {
public:
  Replay(DataConverter converter, int capacity)
      : dataConverter(converter), buffer(ReplayBuffer(capacity)) {}

  void updatePriorities(std::vector<int> indexes, torch::Tensor priorities) {
    for (int i = 0; i < indexes.size(); i++) {
      buffer.update(indexes[i], priorities.index({i}).item<float>());
    }
  }

  inline void putReplayQueue(torch::Tensor priorities,
                             std::vector<ReplayData> data) {
    std::lock_guard<std::mutex> lock(replayMtx);
    if (replayQueue.size() < MAX_REPLAY_QUEUE_SIZE) {
      replayQueue.push_back(std::make_tuple(priorities, data));
    }
  }

  inline void replayDataAdd() {
    while (!replayQueue.empty()) {
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
  }

  SampleData sample() {
    while (buffer.get_count() < REPLAY_BUFFER_MIN_SIZE) {
      replayDataAdd();
    }

    replayDataAdd();
    auto sample = buffer.sample(BATCH_SIZE);
    auto indexes = std::get<0>(sample);
    auto data = dataConverter.toBatchedTrainData(std::get<1>(sample));
    return SampleData(indexes, data);
  }

  DataConverter dataConverter;
  ReplayBuffer buffer;
  std::deque<std::tuple<torch::Tensor, std::vector<ReplayData>>> replayQueue;
  std::mutex replayMtx;
};

struct ReplayDataset
    : torch::data::datasets::StreamDataset<ReplayDataset,
                                           std::vector<SampleData>> {
  Replay &replay;
  ReplayDataset(Replay &replay_) : replay(replay_) {}
  std::vector<SampleData> get_batch(size_t batch_size) override {
    return std::vector<SampleData>({replay.sample()});
  }

  torch::optional<size_t> size() const override { return torch::nullopt; }

  size_t counter = 0;
};

#endif // REPLAY_HPP
