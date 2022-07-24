#ifndef REQUEST_HPP
#define REQUEST_HPP

#include <condition_variable>
#include <mutex>
#include <torch/torch.h>
#include <vector>

struct Event {
  bool notify = false;
  std::mutex mtx;
  std::condition_variable cv;

  void wait() {
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&] { return notify; });
      notify = false;
    }
  }

  void set() {
    // 共有データの更新
    notify = true;
    cv.notify_one();
  }

  void reset() {
    std::lock_guard<std::mutex> lk(mtx);
    // 共有データの更新
    notify = false;
  }
};

struct Request {
  Request() {}
  Request(torch::Tensor state_, float reward_, bool done_)
      : state(state_), reward(reward_), done(done_) {}
  torch::Tensor state;
  float reward;
  bool done;
};

#endif // REQUEST_HPP
