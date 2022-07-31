#include "Learner.hpp"
#include <cstdio>
#include <pwd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

using namespace torch::indexing;

const auto MAX_INFER_COUNT = 16;
std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>>
    gInferOutputs[MAX_INFER_COUNT] = {std::make_tuple(
        torch::zeros({8, ACTION_SIZE}),
        std::make_tuple(torch::zeros({8, 1, 512}), torch::zeros({8, 1, 512})))};

std::atomic<int> gInferOutputCount = 0;

int Learner::listenActor() {

  int ret_code = 0;
  uint32_t buf_len = 0;

  int fd_accept = -1; // 接続受け付け用のFD
  int fd_other = -1;  // sendとかrecv用のFD

  struct passwd *pw = getpwuid(getuid());
  auto home = std::string(pw->pw_dir);
  char INFER_SOCKET[50];
  sprintf(INFER_SOCKET, "%s/infer.sock", home.c_str());

  remove(INFER_SOCKET);

  // ソケットアドレス構造体←今回はここがUNIXドメイン用のやつ
  struct sockaddr_un sun, sun_client;
  memset(&sun, 0, sizeof(sun));
  memset(&sun_client, 0, sizeof(sun_client));

  socklen_t socklen = sizeof(sun_client);

  // UNIXドメインのソケットを作成
  fd_accept = socket(AF_LOCAL, SOCK_STREAM, 0);
  if (fd_accept == -1) {
    printf("failed to socket(errno:%d, error_str:%s)\n", errno,
           strerror(errno));
    return -1;
  }

  // ソケットアドレス構造体を設定
  sun.sun_family = AF_LOCAL;          // UNIXドメイン
  strcpy(sun.sun_path, INFER_SOCKET); // UNIXドメインソケットのパスを指定

  // 上記設定をソケットに紐づける
  ret_code = bind(fd_accept, (const struct sockaddr *)&sun, sizeof(sun));
  if (ret_code == -1) {
    printf("failed to bind(errno:%d, error_str:%s)\n", errno, strerror(errno));
    close(fd_accept);
    return -1;
  }

  // ソケットに接続待ちを設定する。numEnvsはバックログ、同時に何個迄接続要求を受け付けるか。
  ret_code = listen(fd_accept, numEnvs);
  if (ret_code == -1) {
    printf("failed to listen(errno:%d, error_str:%s)\n", errno,
           strerror(errno));
    close(fd_accept);
    return -1;
  }

  std::vector<std::thread> threadList;

  // 無限ループのサーバー処理
  while (1) {
    // printf("accept wating...\n");

    fd_other = accept(fd_accept, (struct sockaddr *)&sun_client, &socklen);
    if (fd_other == -1) {
      printf("failed to accept(errno:%d, error_str:%s)\n", errno,
             strerror(errno));
      continue;
    }

    auto t = std::thread(&Learner::sendAndRecieveActor, this, fd_other);

    threadList.emplace_back(std::move(t));
  }

  for (auto i = 0; i < threadList.size(); i++) {
    threadList[i].join();
  }

  return 0;
}

int Learner::sendAndRecieveActor(int fd_other) {
  ssize_t size = 0;
  uint32_t buf_len = 0;
  char buf[10240];
  Request request;
  int action;
  Event event;

  // 初回のみパケットサイズを得る
  size = recv(fd_other, &buf_len, sizeof(buf_len), 0);
  // printf("buf_len = %d\n", buf_len);

  while (1) {
    // データ本体の受信
    size = recv(fd_other, buf, buf_len, 0);
    // printf("buf_len = %d\n", buf_len);

    int task = buf[0];

    request.state = torch::from_blob(&buf[4], state.sizes(), state.dtype());
    request.reward = buf[buf_len - 5];
    request.done = buf[buf_len - 1];

    action = inference(task, request);

    size = send(fd_other, &action, sizeof(action), 0);
    if (size < 0) {
      perror("send");
    }
  }
}

int Learner::inference(int envId, Request &request) {
  int action;
  std::vector<ReplayData> replayDatas;
  std::vector<RetraceQ> retraceQs;

  InferInput inferInput(state, 1, 1);
  localBuffer.setInferenceParam(envId, request, &inferInput);

  std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>> ret;
  {
    c10::InferenceMode guard(true);
    ret = agent.onlineNet.forward(inferInput.state, inferInput.prevAction,
                                  inferInput.PrevReward);
  }

  auto q = std::get<0>(ret).index({0});
  auto lstmStates = std::get<1>(ret);
  auto ih = std::get<0>(lstmStates).permute({1, 0, 2}).index({0});
  auto hh = std::get<1>(lstmStates).permute({1, 0, 2}).index({0});

  // 選択アクションの確率
  auto policy = std::get<0>(torch::max(torch::softmax(q, 1), 1));

  // std::cout << "policies: " << policies << std::endl;

  // 環境ごとの閾値
  auto epsThreshold =
      torch::pow(0.4, torch::linspace(1., 8., numEnvs)).index({envId});

  auto randomAction = torch::randint(0, q.size(1), 1).to(torch::kLong);
  auto prob = torch::randn(1);

  auto selectAction =
      torch::where(prob < epsThreshold, randomAction, torch::argmax(q, 1));

  localBuffer.updateAndGetTransition(envId, request, selectAction, ih, hh, q,
                                     policy, &replayDatas, &retraceQs);

  if (!replayDatas.empty()) {

    int batchSize = replayDatas.size();
    RetraceData retraceData(batchSize, 1 + TRACE_LENGTH, actionSize);

    dataConverter.toBatchedRetraceData(replayDatas, retraceQs, &retraceData);

    auto priorities = std::get<1>(retraceLoss(
        retraceData.action, retraceData.reward, retraceData.done,
        retraceData.policy, retraceData.onlineQ, retraceData.targetQ));

    replay.putReplayQueue(priorities, replayDatas);
  }

  return selectAction.item<int>();
}

void Learner::trainLoop() {
  int stepsDone = 0;

  auto optimizer = torch::optim::Adam(
      agent.onlineNet.parameters(),
      torch::optim::AdamOptions().lr(LEARNING_RATE).eps(EPSILON));

  std::deque<float> lossList;

  while (1) {
    auto sampleData = replay.sample();
    auto labels = std::get<0>(sampleData);
    auto indexes = std::get<1>(sampleData);
    auto data = std::get<2>(sampleData);

    // std::cout << "state " << data.state.sizes() << ", " << data.state.dtype()
    // << std::endl; std::cout << "action " << data.action.sizes() << ", " <<
    // data.action.dtype() << std::endl; std::cout << "reward " <<
    // data.reward.sizes() << ", " << data.reward.dtype() << std::endl;
    // std::cout << "done " << data.done.sizes() << ", " << data.done.dtype() <<
    // std::endl; std::cout << "ih " << data.ih.sizes() << ", " <<
    // data.ih.dtype() << std::endl; std::cout << "hh " << data.hh.sizes() << ",
    // " << data.hh.dtype() << std::endl; std::cout << "policy " <<
    // data.policy.sizes() << ", " << data.policy.dtype() << std::endl;

    agent.onlineNet.forward(
        data.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)}),
        data.action.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        data.reward.index({Slice(), Slice(0, REPLAY_PERIOD)}), data.ih,
        data.hh);

    agent.targetNet.forward(
        data.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)}),
        data.action.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        data.reward.index({Slice(), Slice(0, REPLAY_PERIOD)}), data.ih,
        data.hh);

    auto onlineRet = agent.onlineNet.forward(
        data.state.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.action.index({Slice(), Slice(REPLAY_PERIOD, -1)}),
        data.reward.index({Slice(), Slice(REPLAY_PERIOD, -1)}));

    auto targetRet = agent.targetNet.forward(
        data.state.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.action.index({Slice(), Slice(REPLAY_PERIOD, -1)}),
        data.reward.index({Slice(), Slice(REPLAY_PERIOD, -1)}));

    auto retraceRet = retraceLoss(
        data.action.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .unsqueeze(2),
        data.reward.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .squeeze(-1),
        data.done.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.policy.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        std::get<0>(onlineRet), std::get<0>(targetRet));

    auto losses = std::get<0>(retraceRet);
    auto priorities = std::get<1>(retraceRet);

    auto loss = torch::sum(losses);
    auto lossValue = loss.item<float>();
    lossList.push_back(lossValue);
    if (lossList.size() > 100) {
      lossList.pop_front();
    }

    // Reset gradients.
    optimizer.zero_grad();
    // Compute gradients of the loss w.r.t. the parameters of our model.
    loss.backward();
    // Update the parameters based on the calculated gradients.
    optimizer.step();

    replay.updatePriorities(labels, indexes, priorities);

    // ターゲットネットワーク更新
    if (stepsDone % TARGET_UPDATE == 0) {
      agent.onlineNet.saveStateDict("model.pt");
      agent.targetNet.loadStateDict("model.pt", "");
    }

    if (stepsDone % 5 == 0) {
      std::cout << "loss = "
                << std::accumulate(lossList.begin(), lossList.end(), 0.0) /
                       lossList.size()
                << ", steps = " << stepsDone << std::endl;
    }

    stepsDone++;
    // std::cout << "stepsDone " << stepsDone << std::endl;
  }
}

int main(void) {
  int ret_code = 0;
  auto stateTensor =
      torch::zeros({1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
  int actionSize = 9;
  int numEnvs = 16;

  Learner learner(stateTensor, actionSize, numEnvs, TRACE_LENGTH, REPLAY_PERIOD,
                  RETURN_TRANSITION_SIZE, REPLAY_BUFFER_SIZE);

  // actorからのリクエスト受付
  auto inferLoop = std::thread(&Learner::listenActor, &learner);

  // optimizer = optim.Adam(agent.online_net.parameters(), lr=LEARNING_RATE,
  // eps=1e-3)

  learner.trainLoop();

  inferLoop.join();

  return EXIT_SUCCESS;
}
