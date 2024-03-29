#ifndef SUM_TREE_HPP
#define SUM_TREE_HPP

#include <vector>
#include "StructuredData.hpp"
#include <memory>
#include <cmath>

class SumTree
{
public:
  SumTree(int capacity_)
      : capacity(capacity_),
        write(0),
        tree(2 * capacity - 1, 0),
        data(capacity)
  {
  }

  int total()
  {
    return tree[0];
  }

  void add(float p, StoredData data_)
  {
    auto idx = write + capacity - 1;

    data[write] = std::move(data_);
    update(idx, p);

    write += 1;
    if (write >= capacity)
    {
      write = 0;
    }
  }

  void update(int idx, float p)
  {
    auto change = p - tree[idx];

    tree[idx] = p;
    propagate(idx, change);
  }

  std::tuple<int, StoredData&> get(float s)
  {
    auto idx = retrieve(0, s);
    auto dataIdx = idx - capacity + 1;
    return {idx, data[dataIdx]};
  }

private:
  void propagate(int idx, float change)
  {
    auto parent = std::floor((idx - 1) / 2);

    tree[parent] += change;

    if (parent != 0)
    {
      propagate(parent, change);
    }
  }

  int retrieve(int idx, float s)
  {
    auto left = 2 * idx + 1;
    auto right = left + 1;

    if (left >= tree.size())
    {
      return idx;
    }

    if (s <= tree[left])
    {
      return retrieve(left, s);
    }
    else
    {
      return retrieve(right, s - tree[left]);
    }
  }

  int capacity;
  int write;
  std::vector<float> tree;
  std::vector<StoredData> data;
};

#endif // SUM_TREE_HPP
