#ifndef REPLAY_HPP
#define REPLAY_HPP

#include "Request.hpp"
#include "SumTree.hpp"
#include "Utils.hpp"
#include <deque>
#include <memory>
#include <mutex>
#include <random>

class ReplayBuffer {
public:
  ReplayBuffer(int capacity) : tree(SumTree(capacity)), count(0) {}

  int get_count() { return count; }

  void update(int idx, float p) { tree.update(idx, p); }

  void add(float p, ReplayData sample) {
    tree.add(p, sample);
    count += 1;
    if (count < REPLAY_BUFFER_MIN_SIZE) {
      if (count % REPLAY_BUFFER_ADD_PRINT_SIZE == 0) {
        std::cout << "Waiting for the replay buffer to fill up. "
                  << "It currently has " << count;
        std::cout << " elements, waiting for at least "
                  << REPLAY_BUFFER_MIN_SIZE << " elements" << std::endl;
      }
    } else if (count == REPLAY_BUFFER_MIN_SIZE) {
        std::cout << "Replay buffer filled up. "
                  << "It currently has " << count << " elements.";
        std::cout << " Start training." << std::endl;
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

      //(idx, p, data)
      idx_list.emplace_back(index);
      data_list.emplace_back(data);
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

  void updatePriorities(torch::Tensor indexes, torch::Tensor priorities) {
    for (int i = 0; i < indexes.size(0); i++) {
      buffer.update(indexes.index({i}).item<int>(),
                    priorities.index({i}).item<float>());
    }
  }

  inline void putReplayQueue(torch::Tensor priorities,
                             std::vector<ReplayData> data) {
    std::lock_guard<std::mutex> lock(replayMtx);
    if (replayQueue.size() < MAX_REPLAY_QUEUE_SIZE) {
      replayQueue.emplace_back(std::move(std::make_tuple(priorities, data)));
    }
  }

  inline void replayDataAdd() {
    while (!replayQueue.empty()) {
      std::tuple<torch::Tensor, std::vector<ReplayData>> queueData;
      {
        std::lock_guard<std::mutex> lock(replayMtx);
        queueData = std::move(replayQueue.front());
        replayQueue.pop_front();
      }

      auto priorities = std::get<0>(queueData);
      auto dataList = std::get<1>(queueData);

      for (int i = 0; i < dataList.size(); i++) {
        buffer.add(priorities.index({i}).item<float>(), dataList[i]);
      }
    }
  }

  auto sample() {
    while (buffer.get_count() < REPLAY_BUFFER_MIN_SIZE) {
      replayDataAdd();
    }

    replayDataAdd();

    auto sample = buffer.sample(BATCH_SIZE);
    auto indexes = std::get<0>(sample);
    auto data = dataConverter.toBatchedTrainData(std::get<1>(sample));
    return std::make_tuple(indexes, data);
  }

  DataConverter dataConverter;
  ReplayBuffer buffer;
  std::deque<std::tuple<torch::Tensor, std::vector<ReplayData>>> replayQueue;
  std::mutex replayMtx;
};

class ReplayDataset : public torch::data::datasets::Dataset<ReplayDataset> {
  using Example = torch::data::Example<>;

private:
  Replay &replay;

public:
  ReplayDataset(Replay &replay_) : replay(replay_){};

  c10::optional<size_t> size() const override {
    return c10::optional<size_t>(1);
  };

  Example get(size_t index) {
    auto data = replay.sample();
    return {torch::tensor(std::get<0>(data)), torch::empty({0})};
  }

  std::vector<Example> get_batch(c10::ArrayRef<size_t> indices) override {
    auto sample = replay.sample();
    auto data = std::get<1>(sample);
    return {Example(torch::tensor(std::get<0>(sample)), data.state),
            Example(data.action, data.reward), Example(data.done, data.ih),
            Example(data.hh, data.policy)};
  }
};

#endif // REPLAY_HPP
