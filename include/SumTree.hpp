#ifndef SUM_TREE_HPP
#define SUM_TREE_HPP

#include <vector>
#include "Utils.hpp"

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

  void add(float p, ReplayData data_)
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

  std::tuple<int, float, ReplayData> get(float s)
  {
    std::cout << "---------------------- " << s << std::endl;
std::cout << "tree.size(): " << tree.size() << std::endl;
    auto idx = retrieve(0, s);
    auto dataIdx = idx - capacity + 1;
    return std::make_tuple(idx, tree[idx], std::move(data[dataIdx]));
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

 std::cout<< "idx: " << idx << ", s: " << s << ", left: " << left << ", right: " << right << std::endl;
 std::cout<< "tree[left]: " << tree[left] << std::endl;
    if (left >= tree.size())
    {
std::cout << "left >= tree.size()" << std::endl;
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
  std::vector<ReplayData> data;
};

#endif // SUM_TREE_HPP
