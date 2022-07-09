#include "Learner.hpp"
#include <cstdio>
#include <pwd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

using namespace torch::indexing;

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

    threadList.push_back(std::move(t));
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
  std::vector<int> taskList;

  // 初回のみパケットサイズを得る
  size = recv(fd_other, &buf_len, sizeof(buf_len), 0);
  // printf("buf_len = %d\n", buf_len);

  while (1) {
    // データ本体の受信
    size = recv(fd_other, buf, buf_len, 0);
    // printf("buf_len = %d\n", buf_len);

    int task = buf[0];
    // if (task == 4) 
    //   std::cout << "recv " << task << " id " << std::this_thread::get_id()  << std::endl;
    reqManager.requests[task].state =
        torch::from_blob(&buf[4], state.sizes(), state.dtype());
    reqManager.requests[task].reward = buf[buf_len - 5];
    reqManager.requests[task].done = buf[buf_len - 1];

    reqManager.addTask(task, &taskList);

    // タスクリストがあるなら、推論を呼ぶ
    if (!taskList.empty()) {
      inference(taskList);
      taskList.clear();
    }

    while (1) {
      if (actions[task] != INVALID_ACTION) {
        break;
      }
    }
    auto action = actions[task];

    actions[task] = INVALID_ACTION;
    // printf("after loop %d\n", task);

    size = send(fd_other, &action, sizeof(action), 0);
    if (size < 0) {
      perror("send");
    }
  }
}

int Learner::inference(std::vector<int> & envIds) {
  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "0 inference " << envIds << std::endl;
  std::vector<Request> &requests = reqManager.requests;
  InferenceData params(state, inferBatchSize, 1);
  localBuffer.setInferenceParam(envIds, &params, requests);

  std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>> forwardRet;

  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "1 inference " << envIds << std::endl;
  {
    c10::InferenceMode guard(true);
    forwardRet =
        agent.onlineNet.forward(params.state, params.prevAction,
                                params.PrevReward);
  }
  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
    // std::cout << "2 inference " << envIds << std::endl;

  auto q = std::get<0>(forwardRet);

  // std::cout << "q: " << q << std::endl;

  auto lstmStates = std::get<1>(forwardRet);

  // 選択アクションの確率
  auto policies = std::get<0>(torch::max(torch::softmax(q, 2), 2));

  // std::cout << "policies: " << policies << std::endl;

  // 環境ごとの閾値
  auto epsThreshold = torch::pow(0.4, torch::linspace(1., 8., requests.size()))
                          .index({torch::tensor(envIds)});

  auto randomActions =
      torch::randint(0, q.size(2), envIds.size()).to(torch::kLong);
  auto probs = torch::randn(envIds.size());

  // std::cout << "probs: " << probs<< std::endl;
  // std::cout << "epsThreshold: " << epsThreshold << std::endl;
  // std::cout << "randomActions: " << randomActions << std::endl;

  auto selectActions = torch::where(probs < epsThreshold, randomActions,
                                    torch::argmax(q, 2).squeeze(1));
  
  assert(randomActions.size(0) != 0);
  assert(q.size(0) != 0);
  assert(torch::argmax(q, 2).squeeze(1).size(0) != 0);
  assert(selectActions.size(0) != 0);


  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "3 inference " << envIds << std::endl;
  auto transitions = localBuffer.updateAndGetTransitions(
      envIds, requests, selectActions, lstmStates, q, policies);

  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "4 inference " << envIds << std::endl;
  auto replayDatas = std::get<0>(transitions);
  if (replayDatas.size() > 0) {
    auto retraceData = dataConverter.toBatchedRetraceData(
        replayDatas, std::get<1>(transitions));
    auto priorities = std::get<1>(retraceLoss(
        retraceData.action, retraceData.reward, retraceData.done,
        retraceData.policy, retraceData.onlineQ, retraceData.targetQ));
  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "5 inference " << envIds << std::endl;

  //   std::cout << "$$$$$$$$$$$$$$$ replayDatas[0].state.sizes " << replayDatas[0].state.sizes() << std::endl;

    for(auto r: replayDatas) {
      assert(r.action.size(0) != 0);
    }
    replay.putReplayQueue(priorities, replayDatas);
  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "6 inference " << envIds << std::endl;
  }

  // if(std::find(envIds.begin(), envIds.end(), 4) != envIds.end())
  //   std::cout << "7 inference " << envIds << std::endl;
  for (int i = 0; i < envIds.size(); i++) {
    actions[envIds[i]] = selectActions.index({i}).item<int>();
  }

  return 1;
}

void Learner::trainLoop() {
  int stepsDone = 0;

  torch::optim::Adam optimizer(agent.onlineNet.parameters(), /*lr=*/0.01);

  while (1) {
    auto sample = replay.sample();
    auto indexes = std::get<0>(sample);
    auto data = std::get<1>(sample);

    std::cout << "state " << data.state.sizes() << ", " << data.state.dtype() << std::endl;
    std::cout << "action " << data.action.sizes() << ", " << data.action.dtype() << std::endl;
    std::cout << "reward " << data.reward.sizes() << ", " << data.reward.dtype() << std::endl;
    std::cout << "done " << data.done.sizes() << ", " << data.done.dtype() << std::endl;
    std::cout << "ih " << data.ih.sizes() << ", " << data.ih.dtype() << std::endl;
    std::cout << "hh " << data.hh.sizes() << ", " << data.hh.dtype() << std::endl;
    std::cout << "policy " << data.policy.sizes() << ", " << data.policy.dtype() << std::endl;

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
        data.action.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.reward.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.done.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.policy.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        std::get<0>(onlineRet), std::get<0>(targetRet));

    auto losses = std::get<0>(retraceRet);
    auto priorities = std::get<0>(retraceRet);

    // Reset gradients.
    optimizer.zero_grad();
    // Compute gradients of the loss w.r.t. the parameters of our model.
    torch::sum(losses).backward();
    // Update the parameters based on the calculated gradients.
    optimizer.step();

    replay.putPriorityQueue(indexes, priorities);

    // ターゲットネットワーク更新
    if (stepsDone % TARGET_UPDATE == 0) {
      agent.onlineNet.saveStateDict("model.pt");
      agent.targetNet.loadStateDict("model.pt", "");
    }

    stepsDone++;
  }
}

int main(void) {
  int ret_code = 0;
  auto stateTensor =
      torch::zeros({1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
  int actionSize = 9;
  int numEnvs = 16;
  int traceLength = 80;
  int replayPeriod = 40;
  int returnSize = 32;
  int capacity = 25000;

  Learner learner(actionSize, stateTensor, numEnvs, traceLength, replayPeriod,
                  returnSize, capacity);

  // actorからのリクエスト受付
  auto inferLoop = std::thread(&Learner::listenActor, &learner);

  // リプレイループ
  auto replayLoop = std::thread(&Replay::loop, learner.getReplay());

  learner.trainLoop();

  inferLoop.join();
  replayLoop.join();

  return EXIT_SUCCESS;
}
