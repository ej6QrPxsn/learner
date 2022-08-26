#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include "Request.hpp"
#include "SumTree.hpp"
#include "Utils.hpp"
#include <memory>
#include <mutex>
#include <random>

class ReplayBuffer {
public:
  ReplayBuffer(int capacity) : tree(SumTree(capacity)), count(0) {}

  int get_count() { return count; }

  void update(int idx, float p) {
    std::lock_guard<std::mutex> lock(mtx);
    tree.update(idx, p);
  }

  void add(float p, ReplayData sample) {
    std::lock_guard<std::mutex> lock(mtx);
    tree.add(p, compress(sample));
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
    std::random_device rd;
    std::default_random_engine eng(rd());

    auto segment = tree.total() / n;

    for (auto i = 0; i < n; i++) {
      auto a = segment * i;
      auto b = segment * (i + 1);

      std::uniform_int_distribution<int> distr(a, b);

      auto s = distr(eng);
      if (s == 0) {
        s = 1;
      }
      auto ret = tree.get(s);
      auto index = std::get<0>(ret);
      auto & data = std::get<1>(ret);

      //(idx, p, data)
      idx_list.emplace_back(index);
      data_list.emplace_back(decompress(data));
    }

    return {idx_list, data_list};
  }

private:
  SumTree tree;
  int count;
  std::mutex mtx;
};

#endif // REPLAY_BUFFER_HPP
