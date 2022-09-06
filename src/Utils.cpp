#include "Utils.hpp"
#include "Models.hpp"
#include <future>
#include <zstd.h> // presumes zstd library is installed

using namespace torch::indexing;

std::array<torch::OrderedDict<std::string, at::Tensor>, NUM_TRAIN_THREADS>
    gParams;
std::array<Event, NUM_TRAIN_THREADS> gEvents;
torch::OrderedDict<std::string, at::Tensor> gTotalPrarams;
std::array<bool, NUM_TRAIN_THREADS> gGetGrads;

inline auto h(torch::Tensor x) {
  auto eps = RESCALING_EPSILON;
  return torch::sign(x) * (torch::sqrt(torch::abs(x) + 1.) - 1.) + eps * x;
}

inline auto h_1(torch::Tensor x) {
  auto eps = RESCALING_EPSILON;
  return torch::sign(x) *
         (torch::sqrt(
              ((torch::sqrt(1. + 4. * eps * (torch::abs(x) + 1. + eps))) - 1.) /
              (2. * eps)) -
          1.);
}

inline auto getRetraceOperatorSigma(int s, torch::Tensor td,
                                    torch::Tensor retraceCoefficients) {
  auto tdValue = td.index({Slice(), s}).unsqueeze(1);
  // std::cout << "tdValue: " << tdValue.sizes() << std::endl;
  std::vector<torch::Tensor> values;
  for (int j = s; j < TRACE_LENGTH; j++) {
    auto val =
        std::pow(DISCOUNT_GAMMA, j - s) *
        torch::prod(retraceCoefficients.index({Slice(), Slice(s + 1, j + 1)})) *
        tdValue;
    // std::cout << "else: " << val.sizes() << std::endl;
    values.emplace_back(val);
    // }
  }

  //  std::cout << "values: " << values[0].sizes() << std::endl;
  // seq, batch
  auto sequenceValues = torch::cat(values, 1);

  // batch
  return torch::sum(sequenceValues, 1).unsqueeze(1);
}

std::tuple<float, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ, torch::Tensor targetQ,
            torch::Device device, torch::optim::Adam *optimizer) {
  auto batchSize = action.size(0);
  auto retraceLength = action.size(1) - 1;

  // std::cout << "action: " << action.sizes() << std::endl;
  // std::cout << "reward: " << reward.sizes() << std::endl;
  // std::cout << "done: " << done.sizes() << std::endl;
  // std::cout << "policy: " << policy.sizes() << std::endl;
  // std::cout << "onlineQ: " << onlineQ.sizes() << std::endl;

  // オンラインポリシーのgreedyなものがターゲットポリシー
  auto targetPolicy = torch::amax(torch::softmax(onlineQ, 2), 2);
  // std::cout << "targetPolicy: " << targetPolicy.sizes() << std::endl;

  auto currentTargetQValue =
      torch::amax(targetQ.index({Slice(), Slice(None, -1)}), 2);
  // std::cout << "currentTargetQValue: " << currentTargetQValue.sizes() <<
  // std::endl;

  auto nextTargetPolicy =
      torch::softmax(targetQ.index({Slice(), Slice(1, None)}), 2);
  // std::cout << "nextTargetPolicy: " << nextTargetPolicy.sizes() << std::endl;
  // std::cout << "targetQ: " << targetQ.index({Slice(), Slice(1,
  // None)}).sizes() << std::endl; std::cout << "torch::sum: " << torch::sum(
  // h_1(nextTargetPolicy * targetQ.index({Slice(), Slice(1, None)})),
  // 2).sizes() << std::endl;

  auto nextTargetQValue =
      DISCOUNT_GAMMA *
      torch::sum(
          h_1(nextTargetPolicy * targetQ.index({Slice(), Slice(1, None)})), 2);

  // std::cout << "reward.index({Slice(), Slice(None, -1)}): " <<
  // reward.index({Slice(), Slice(None, -1)}).sizes() << std::endl; std::cout <<
  // "nextTargetQValue: " << nextTargetQValue.sizes() << std::endl; std::cout <<
  // "currentTargetQValue: " << currentTargetQValue.sizes() << std::endl;
  auto td = reward.index({Slice(), Slice(None, -1)}) + nextTargetQValue -
            currentTargetQValue;
  // std::cout << "td: " << td.sizes() << std::endl;

  // retrace coefficients
  // ゼロ除算防止
  auto zero = torch::zeros({1}).to(device);
  auto one = torch::ones({1}).to(device);
  auto epsilon = torch::empty({1}).index_put_({0}, 1e-6).to(device);

  auto noZeroPolicy = torch::where(policy == zero, epsilon, policy);
  auto retraceCoefficients =
      RETRACE_LAMBDA * torch::minimum(targetPolicy / noZeroPolicy, one);
  // std::cout << "retraceCoefficients: " << retraceCoefficients.sizes() <<
  // std::endl;

  // batch, seqごとのリトレースオペレーターの中のシグマ配列
  // batch, seq <- [batch, 1] * seq
  std::vector<torch::Tensor> sigmaList;
  for (auto s = 0; s < retraceLength; s++) {
    sigmaList.emplace_back(getRetraceOperatorSigma(s, td, retraceCoefficients));
  }

  auto retraceOperatorSigma = torch::cat(sigmaList, 1);
  // std::cout << "retraceOperatorSigma: " << retraceOperatorSigma.sizes() <<
  // std::endl;

  auto retraceOperator = h(h_1(currentTargetQValue) + retraceOperatorSigma);
  // std::cout << "retraceOperator: " << retraceOperator.sizes() << std::endl;

  // std::cout << "onlineQ: " << onlineQ.sizes() << std::endl;
  // std::cout << "action: " << action.sizes() << std::endl;
  // t時点のオンラインネットのアクションのQ値
  // batch, seq, actions
  auto qValue = onlineQ.index({Slice(), Slice(None, -1)})
                    .gather(2, action.index({Slice(), Slice(None, -1)}));
  // std::cout << "qValue: " << qValue.sizes() << std::endl;

  // batch, seq
  auto absErrors = torch::abs(qValue.squeeze(2) - retraceOperator);
  // std::cout << "absErrors: " << absErrors.sizes() << std::endl;

  // batch <- batch, seq
  auto priorities =
      (ETA * torch::amax(absErrors, 1) + (1 - ETA) * torch::mean(absErrors, 1));
  // std::cout << "priorities: " << priorities.sizes() << std::endl;

  // batch <- batch, seq
  auto losses = torch::sum(torch::square(absErrors), 1);
  // std::cout << "losses: " << losses.sizes() << std::endl;

  auto loss = torch::mean(losses);

  if (optimizer != nullptr) {
    // Reset gradients.
    optimizer->zero_grad();
    // Compute gradients of the loss w.r.t. the parameters of our model.
    loss.backward();
  }

  return {loss.item<float>(), priorities.detach()};
}

StoredData compress(ReplayData &replayData) {
  char tmp[sizeof(ReplayData)];

  size_t const maxCompressedSize = ZSTD_compressBound(sizeof(ReplayData));
  size_t const compressedSize =
      ZSTD_compress(tmp, maxCompressedSize, &replayData, sizeof(ReplayData),
                    ZSTD_CLEVEL_DEFAULT);
  auto code = ZSTD_isError(compressedSize);
  if (code) {
    exit(code);
  }

  StoredData data;
  data.size = compressedSize;
  data.ptr = std::unique_ptr<char[]>(new char[compressedSize]);
  memcpy(data.ptr.get(), tmp, compressedSize);
  return std::move(data);
}

void decompress(StoredData &compressed, ReplayData &replayData) {
  size_t const decompressedSize = ZSTD_decompress(
      &replayData, sizeof(ReplayData), compressed.ptr.get(), compressed.size);
  auto code = ZSTD_isError(decompressedSize);
  if (code) {
    exit(code);
  }
}

void updateGrad(int threadNum, R2D2Agent &model) {

  auto currentParams = model.named_parameters(true /*recurse*/);
  gGetGrads[threadNum] = true;

  if (std::count(gGetGrads.begin(), gGetGrads.end(), true) ==
      NUM_TRAIN_THREADS) {
    gTotalPrarams = currentParams;

    // すべてのパラメーター
    for (auto &val : gTotalPrarams) {
      torch::Tensor tmpGradItem;
      auto name = val.key();
      auto *totalGradItem = gTotalPrarams.find(name);
      if (totalGradItem != nullptr) {
        tmpGradItem = totalGradItem->grad();
      }

      // std::cout << name << ": " << totalGrad.index({0, 0}) << std::endl;

      // 自分ベースの変数に自分以外のスレッドの勾配を足す
      for (int i = 0; i < NUM_TRAIN_THREADS; i++) {
        if (i != threadNum) {
          auto *t = gParams[i].find(name);
          if (t != nullptr) {
            // std::cout << thread << ": " << t->grad().index({0, 0}) <<
            // std::endl;
            tmpGradItem += t->grad();
          }
        }
      }
      // std::cout << "totalGrad: " << totalGrad.index({0, 0}) << std::endl;

      // 集計した勾配をグローバル変数に設定
      totalGradItem->mutable_grad() = tmpGradItem;
    }

    // 勾配取得状態クリア
    std::fill(gGetGrads.begin(), gGetGrads.end(), false);

    // 他スレッド再開
    for (int i = 0; i < NUM_TRAIN_THREADS; i++) {
      if (i != threadNum) {
        gEvents[i].set();
      }
    }

  } else {
    gParams[threadNum] = currentParams;

    // 集計が終わるまで待つ
    gEvents[threadNum].wait();
  }

  // 集計した勾配をすべてのパラメーターに設定する
  for (auto &gVal : gTotalPrarams) {
    auto name = gVal.key();
    auto *currentItem = currentParams.find(name);
    if (currentItem != nullptr) {
      currentItem->mutable_grad() = gVal.value().grad();
    }
  }
}
