#include "Utils.hpp"
#include "Common.hpp"

using namespace torch::indexing;

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
                                    torch::Tensor retraceCoefficients,
                                    int batchSize) {
  auto tdValue = td.index({Slice(), s}).unsqueeze(1);
  // std::cout << "tdValue: " << tdValue.sizes() << std::endl;
  std::vector<torch::Tensor> values;
  for (int j = s; j < TRACE_LENGTH; j++) {
  //   if (j < 2) {
  //     auto val = torch::ones({batchSize, 1});
  // std::cout << "j < 2: " << val.sizes() << std::endl;
  //     values.push_back(val);
  //   } else {
  // std::cout << "else1: " << retraceCoefficients.sizes() << std::endl;
  // std::cout << "else2: " << retraceCoefficients.index(
  //                                      {Slice(), Slice(s + 1, j + 1)}).sizes() << std::endl;
  // std::cout << "else3: " << torch::prod(retraceCoefficients.index(
  //                                      {Slice(), Slice(s + 1, j + 1)}) *
  //                                  tdValue, 1) << std::endl;
  // std::cout << "else4: " << std::pow(DISCOUNT_GAMMA, j - s) << std::endl;
      auto val = std::pow(DISCOUNT_GAMMA, j - s) *
                       torch::prod(retraceCoefficients.index(
                                       {Slice(), Slice(s + 1, j + 1)})) *
                                   tdValue;
  // std::cout << "else: " << val.sizes() << std::endl;
      values.push_back(val);
    // }
  }

  //  std::cout << "values: " << values[0].sizes() << std::endl;
 // seq, batch
  auto sequenceValues = torch::cat(values, 1);

  // batch
  return torch::sum(sequenceValues, 1).unsqueeze(1);
}

std::tuple<torch::Tensor, torch::Tensor>
retraceLoss(torch::Tensor action, torch::Tensor reward, torch::Tensor done,
            torch::Tensor policy, torch::Tensor onlineQ,
            torch::Tensor targetQ) {
  auto batchSize = action.size(0);
  auto retraceLength = action.size(1) - 1;

  // std::cout << "action: " << action.sizes() << std::endl;
  // std::cout << "reward: " << reward.sizes() << std::endl;
  // std::cout << "done: " << done.sizes() << std::endl;
  // std::cout << "policy: " << policy.sizes() << std::endl;
  // std::cout << "onlineQ: " << onlineQ.sizes() << std::endl;


  // オンラインポリシーのgreedyなものがターゲットポリシー
  auto targetPolicy = std::get<0>(torch::max(torch::softmax(onlineQ, 2), 2));
  // std::cout << "targetPolicy: " << targetPolicy.sizes() << std::endl;

  auto currentTargetQValue = std::get<0>(
      torch::max(targetQ.index({Slice(), Slice(None, -1)}), 2));
  // std::cout << "currentTargetQValue: " << currentTargetQValue.sizes() << std::endl;

  auto nextTargetPolicy =
      torch::softmax(targetQ.index({Slice(), Slice(1, None)}), 2);
  // std::cout << "nextTargetPolicy: " << nextTargetPolicy.sizes() << std::endl;
  // std::cout << "targetQ: " << targetQ.index({Slice(), Slice(1, None)}).sizes() << std::endl;
  // std::cout << "torch::sum: " << torch::sum(
          // h_1(nextTargetPolicy * targetQ.index({Slice(), Slice(1, None)})), 2).sizes() << std::endl;

  auto nextTargetQValue =
      DISCOUNT_GAMMA *
      torch::sum(
          h_1(nextTargetPolicy * targetQ.index({Slice(), Slice(1, None)})), 2);

  // std::cout << "reward.index({Slice(), Slice(None, -1)}): " << reward.index({Slice(), Slice(None, -1)}).sizes() << std::endl;
  // std::cout << "nextTargetQValue: " << nextTargetQValue.sizes() << std::endl;
  // std::cout << "currentTargetQValue: " << currentTargetQValue.sizes() << std::endl;
  auto td = reward.index({Slice(), Slice(None, -1)}) + nextTargetQValue -
            currentTargetQValue;
  // std::cout << "td: " << td.sizes() << std::endl;

  // retrace coefficients
  // ゼロ除算防止
  auto zero = torch::zeros({1});
  auto one = torch::ones({1});
  auto epsilon = torch::empty({1}).index_put_({0}, 1e-6);

  auto noZeroPolicy = torch::where(policy == zero, epsilon, policy);
  auto retraceCoefficients =
      RETRACE_LAMBDA * torch::minimum(targetPolicy / noZeroPolicy, one);
  // std::cout << "retraceCoefficients: " << retraceCoefficients.sizes() << std::endl;

  // batch, seqごとのリトレースオペレーターの中のシグマ配列
  // batch, seq <- [batch, 1] * seq
  std::vector<torch::Tensor> sigmaList;
  for (auto s = 0; s < retraceLength; s++) {
    sigmaList.push_back(
        getRetraceOperatorSigma(s, td, retraceCoefficients, batchSize));
  }

  auto retraceOperatorSigma = torch::cat(sigmaList, 1);
  // std::cout << "retraceOperatorSigma: " << retraceOperatorSigma.sizes() << std::endl;

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
  auto priorities = (ETA * std::get<0>(torch::max(absErrors, 1)) +
                     (1 - ETA) * torch::mean(absErrors, 1));
  // std::cout << "priorities: " << priorities.sizes() << std::endl;

  // batch <- batch, seq
  auto losses = torch::sum(torch::square(absErrors), 1);
  // std::cout << "losses: " << losses.sizes() << std::endl;

  return std::make_tuple(losses, priorities);
}