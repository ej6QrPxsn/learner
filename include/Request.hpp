#ifndef REQUEST_HPP
#define REQUEST_HPP

#include <condition_variable>
#include <deque>
#include <mutex>
#include <torch/torch.h>
#include <vector>

struct Event {
  bool notify = false;
  std::mutex mtx;
  std::condition_variable cv;
  int waitCount = 0;

  void wait() {
    {
      std::unique_lock<std::mutex> lk(mtx);
      waitCount++;
      cv.wait(lk, [&] { return notify; });
      if (--waitCount == 0) {
        notify = false;
      }
    }
  }

  bool isReady() {
    return waitCount == 0;
  }

  void set() {
    std::lock_guard<std::mutex> lk(mtx);
    // 共有データの更新
    notify = true;
    cv.notify_all();
  }
};

struct Request {
  Request(torch::Tensor state_, float reward_, bool done_)
      : state(state_), reward(reward_), done(done_) {}
  torch::Tensor state;
  float reward;
  bool done;
};

struct RequestManager {
  std::vector<int> taskList;
  std::vector<Request> requests;
  std::mutex taskMtx;
  std::mutex eventMtx;
  int inferBatchSize;
  int currentAction;

  RequestManager(int numEnvs, int inferBatchSize_, torch::Tensor &state)
      : inferBatchSize(inferBatchSize_), currentAction(0),
        requests(numEnvs, Request(state, 0, 0)) {
    int count = numEnvs / inferBatchSize + 2;
  }

  void addTask(int task, std::vector<int> * tasks) {
    std::lock_guard<std::mutex> lock(taskMtx);
    taskList.push_back(task);

    if (taskList.size() == inferBatchSize) {
      std::copy(taskList.begin(), taskList.end(), std::back_inserter(*tasks));
      taskList.clear();
    }
  }
};

#endif // REQUEST_HPP
