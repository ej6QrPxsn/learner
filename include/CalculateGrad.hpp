#ifndef CALCULATE_GRAD_HPP
#define CALCULATE_GRAD_HPP

#include "StructuredData.hpp"
#include <torch/torch.h>

NamedParameters gTotalGrads;
std::array<Event, NUM_TRAIN_THREADS> gGradEvents;
std::unordered_map<std::string, std::unique_ptr<std::mutex>> gGradLocks;
std::mutex gGradMutex;
int gCalculateCount = 0;

void initTotalGrad(NamedParameters srcParams, torch::Device device) {
  for (auto &val : srcParams) {
    auto name = val.key();
    auto sizes = val.value().sizes();
    gTotalGrads.insert(name, torch::zeros({sizes}).to(device));
    gGradLocks.emplace(name, std::make_unique<std::mutex>());
  }
}

void clearTotalGrad() {
  for (auto &val : gTotalGrads) {
    val.value().zero_();
  }
}

void updateGrad(R2D2Agent &model, int threadNum) {
  auto currentParams = model.named_parameters(true /*recurse*/);
  // すべてのパラメーター
  for (auto &gVal : gTotalGrads) {
    auto name = gVal.key();
    auto &totalValue = gVal.value();
    auto *t = currentParams.find(name);
    // std::cout << threadNum << ": " << name << ": " << t->grad().sizes()
    //           << std::endl;
    // std::cout << totalValue.sizes() << std::endl;
    // std::cout << (totalValue + t->grad()).sizes() << std::endl;

    // この名前のパラメータをロックしてgradを合計に足す
    std::lock_guard<std::mutex> lock(*gGradLocks[name]);
    totalValue += t->grad();
  }

  int count;
  {
    std::lock_guard<std::mutex> lock(gGradMutex);
    count = ++gCalculateCount;
  }

  if (count < NUM_TRAIN_THREADS) {
    gGradEvents[threadNum].wait();
  } else {
    // 他スレッド再開
    for (int i = 0; i < NUM_TRAIN_THREADS; i++) {
      if (i != threadNum) {
        gGradEvents[i].set();
      }
    }
  }

  // 集計した勾配をすべてのパラメーターに設定する
  for (auto &gVal : gTotalGrads) {
    auto name = gVal.key();
    auto *currentItem = currentParams.find(name);
    if (currentItem != nullptr) {
      currentItem->mutable_grad().index_put_({torch::indexing::Slice()},
                                             gVal.value());
    }
  }

  // 全てのスレッドの勾配設定が終わったら、合計勾配値をクリアする
  std::lock_guard<std::mutex> lock(gGradMutex);
  if (--gCalculateCount == 0) {
    clearTotalGrad();
  }
}

#endif // CALCULATE_GRAD_HPP
