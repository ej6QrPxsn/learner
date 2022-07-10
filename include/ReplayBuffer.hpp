#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include "Common.hpp"
#include "SumTree.hpp"
#include <iostream>
#include <random>

class ReplayBuffer {
public:
  ReplayBuffer(int capacity) : tree(SumTree(capacity)), count(0) {}

  int get_count() { return count; }

  void update(int idx, float p) { tree.update(idx, p); }

  void add(float p, ReplayData sample) {
    assert (sample.action.size(0) != 0);
    tree.add(p, sample);
    count += 1;
    if (count < REPLAY_BUFFER_MIN_SIZE) {
      if (count % REPLAY_BUFFER_ADD_PRINT_SIZE == 0) {
        std::cout << "Waiting for the replay buffer to fill up. "
                  << "It currently has " << count;
        std::cout << " elements, waiting for at least " << REPLAY_BUFFER_MIN_SIZE
                  << " elements" << std::endl;
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

      assert (data.action.size(0) != 0);

      //(idx, p, data)
      idx_list.push_back(index);
      data_list.push_back(data);
    }

    return std::move(std::make_tuple(idx_list, data_list));
  }

private:
  SumTree tree;
  int count;
};

#endif // REPLAY_BUFFER_HPP
