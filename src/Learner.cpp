#include "Learner.hpp"
#include "CalculateGrad.hpp"
#include <cstdio>
#include <filesystem>
#include <pwd.h>
#include <shared_mutex>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
// #include <tensorflow/core/util/events_writer.h>

using namespace torch::indexing;
std::vector<Agent> gAgents(NUM_TRAIN_THREADS, Agent(ACTION_SIZE));
std::array<SampleData, NUM_TRAIN_THREADS> gSamples;
std::array<TrainData, NUM_TRAIN_THREADS> gTrainDatas;
NamedParameters gTrainParams;
NamedParameters gTrainBuffers;
int gTrainCount = 0;

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

  R2D2Agent inferModel(1, ACTION_SIZE);
  // 推論モデルの計算グラフは切っておく
  inferModel.detach_();

  // 無限ループのサーバー処理
  while (1) {
    // printf("accept wating...\n");

    fd_other = accept(fd_accept, (struct sockaddr *)&sun_client, &socklen);
    if (fd_other == -1) {
      printf("failed to accept(errno:%d, error_str:%s)\n", errno,
             strerror(errno));
      continue;
    }

    auto t =
        std::thread(&Learner::sendAndRecieveActor, this, fd_other, inferModel);

    threadList.emplace_back(std::move(t));
  }

  for (auto i = 0; i < threadList.size(); i++) {
    threadList[i].join();
  }

  return 0;
}

int Learner::sendAndRecieveActor(int fd_other, R2D2Agent inferModel) {
  size_t size;
  Request request;
  int action;
  int envId;
  int steps = 0;
  int prevTrainCount = 0;

  torch::Device device(torch::kCPU);

  LocalBuffer localBuffer(state, numEnvs, device);
  AgentInput agentInput(state, 1, 1, device);

  // アクター番号
  size = recv(fd_other, &envId, sizeof(envId), 0);

  while (1) {
    // データ本体の受信
    size = recv(fd_other, &request, sizeof(request), 0);

    action = inference(inferModel, request, agentInput, device, localBuffer);

    size = send(fd_other, &action, sizeof(action), 0);
    if (size < 0) {
      perror("send");
    }

    steps++;

    if (gTrainCount != prevTrainCount && steps % 100 == 0 && envId == 0) {
      prevTrainCount = gTrainCount;
      inferModel.copyParams(gTrainParams, gTrainBuffers);
    }
  }
}

int Learner::inference(R2D2Agent &inferModel, Request &request,
                       AgentInput &agentInput, torch::Device device,
                       LocalBuffer &localBuffer) {
  int action;

  localBuffer.setInferenceParam(request, &agentInput);

  AgentOutput out = inferModel.forward(
      agentInput.state, agentInput.prevAction, agentInput.prevReward,
      LstmStates(agentInput.hiddenStates, agentInput.cellStates), device);

  auto q = std::get<0>(out);

  // 選択アクションの確率
  auto policy = torch::amax(torch::softmax(q, 2), 2);

  // std::cout << "policies: " << policies << std::endl;

  // 環境ごとの閾値
  auto epsThreshold =
      torch::pow(0.4, torch::linspace(1., 8., numEnvs)).index({request.envId});

  auto randomAction = torch::randint(0, q.size(2), 1).to(torch::kLong);
  auto prob = torch::randn(1);

  auto selectAction =
      torch::where(prob < epsThreshold, randomAction, torch::argmax(q, 2));

  auto ret = localBuffer.updateAndGetTransition(request, selectAction, q,
                                                std::get<1>(out), policy);

  if (ret) {
    auto &retraceData = localBuffer.getRetraceData();
    auto priorities = std::get<1>(retraceLoss(
        retraceData.action, retraceData.reward, retraceData.done,
        retraceData.policy, retraceData.onlineQ, retraceData.targetQ, device));

    replay.putReplayQueue(priorities, std::move(localBuffer.getReplayData()));
  }

  return selectAction.item<int>();
}

void Learner::trainLoop(int threadNum) {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  int stepsDone = 0;
  // std::string envent_file = "/home/yoshi/logs";
  // tensorflow::EventsWriter writer(envent_file);

  Agent &agent = gAgents[threadNum];

  auto optimizer = torch::optim::Adam(
      agent.onlineNet.parameters(),
      torch::optim::AdamOptions().lr(LEARNING_RATE).eps(EPSILON));

  auto &sampleData = gSamples[threadNum];
  auto &trainData = gTrainDatas[threadNum];

  std::deque<float> lossList;

  if (threadNum == 0) {
    initTotalGrad(agent.onlineNet.named_parameters(), device);
  }

  while (1) {
    replay.sample(sampleData);
    toBatchedTrainData(trainData, sampleData.dataList);

    // // モデルに設定するhidden stateをdetach
    trainData.hiddenStates.detach_();
    trainData.cellStates.detach_();

    // 計算グラフを切る
    agent.onlineNet.detach_();

    // Reset gradients.
    optimizer.zero_grad();

    auto burnInOnlineRet = agent.onlineNet.forward(
        trainData.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)}),
        trainData.action.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        trainData.reward.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        LstmStates(trainData.hiddenStates, trainData.cellStates), device);
    auto [onlineHiddenStates, onlineCellStates] = std::get<1>(burnInOnlineRet);

    auto burnInOnTargetRet = agent.targetNet.forward(
        trainData.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)}),
        trainData.action.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        trainData.reward.index({Slice(), Slice(0, REPLAY_PERIOD)}),
        LstmStates(trainData.hiddenStates, trainData.cellStates), device);

    // ここから勾配を使う
    // backwardではここまでさかのぼる
    onlineHiddenStates.requires_grad_(true);
    onlineCellStates.requires_grad_(true);
    agent.onlineNet.requiresGrad_(true);

    auto onlineRet = agent.onlineNet.forward(
        trainData.state.index({Slice(), Slice(REPLAY_PERIOD, None)}),
        trainData.action.index({Slice(), Slice(REPLAY_PERIOD - 1, -1)}),
        trainData.reward.index({Slice(), Slice(REPLAY_PERIOD - 1, -1)}),
        LstmStates(onlineHiddenStates, onlineCellStates), device);

    auto targetRet = agent.targetNet.forward(
        trainData.state.index({Slice(), Slice(REPLAY_PERIOD, None)}),
        trainData.action.index({Slice(), Slice(REPLAY_PERIOD - 1, -1)}),
        trainData.reward.index({Slice(), Slice(REPLAY_PERIOD - 1, -1)}),
        std::get<1>(burnInOnTargetRet), device);

    auto [loss, priorities] = retraceLoss(
        trainData.action.index({Slice(), Slice(REPLAY_PERIOD, None)})
            .unsqueeze(2),
        trainData.reward.index({Slice(), Slice(REPLAY_PERIOD, None)})
            .squeeze(-1),
        trainData.done.index({Slice(), Slice(REPLAY_PERIOD, None)}),
        trainData.policy.index({Slice(), Slice(REPLAY_PERIOD, None)}),
        std::get<0>(onlineRet), std::get<0>(targetRet), device, true);

    // std::cout << "----------------------" << std::endl;
    // for (auto &val : agent.onlineNet.named_parameters()) {
    //   auto name = val.key();
    //   if (val.value().grad().numel() > 0) {
    //     std::cout << "1"
    //               << ": " << name << ": "
    //               << val.value().grad().reshape(-1).index({-1}) <<
    // std::endl;
    //   } else {
    //     std::cout << "1"
    //               << ": " << name << ": " << std::endl;
    //   }
    // }

    // 勾配を集計して設定
    updateGrad(agent.onlineNet, threadNum);

    // Update the parameters based on the calculated gradients.
    optimizer.step();

    if (threadNum == 0) {
      gTrainParams = agent.onlineNet.named_parameters(true);
      gTrainBuffers = agent.onlineNet.named_buffers(true);
      agent.trainCount++;

      lossList.push_back(loss);
      if (lossList.size() > 100) {
        lossList.pop_front();
      }
    }

    replay.updatePriorities(sampleData.labelList, sampleData.indexList,
                            priorities);

    // tensorflow::Event event;
    // tensorflow::Summary::Value* summ_val =
    // event.mutable_summary()->add_value(); event.set_step(stepsDone);
    // summ_val->set_tag("loss");
    // summ_val->set_simple_value(loss);
    // writer->WriteEvent(event);

    stepsDone++;

    // ターゲットネットワーク更新
    if (stepsDone % TARGET_UPDATE == 0) {
      agent.targetNet.copyFrom(agent.onlineNet);
    }

    if (threadNum == 0 /* && (stepsDone % 5 == 0)*/) {

      std::cout << "loss = "
                << std::accumulate(lossList.begin(), lossList.end(), 0.0) /
                       lossList.size()
                << ", steps = " << stepsDone * NUM_TRAIN_THREADS << std::endl;
    }

    // モデル保存
    if (threadNum == 0 && (stepsDone % 1000 == 0)) {
      agent.onlineNet.saveStateDict("model.pt");
    }
    std::cout << "stepsDone " << stepsDone << std::endl;
  }
}

int main(void) {
  int ret_code = 0;
  auto stateTensor =
      torch::zeros({1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
  int actionSize = 9;
  int numEnvs = NUM_ENVS;

  // 訓練モデルのパラメーターを合わせる
  for (auto i = 1; i < NUM_TRAIN_THREADS; i++) {
    gAgents[i].onlineNet.copyFrom(gAgents[0].onlineNet);
    gAgents[i].targetNet.copyFrom(gAgents[0].targetNet);
  }

  Learner learner(stateTensor, actionSize, numEnvs, TRACE_LENGTH, REPLAY_PERIOD,
                  REPLAY_BUFFER_SIZE);

  // actorからのリクエスト受付
  auto inferThread = std::thread(&Learner::listenActor, &learner);

  inferThread.join();

  return EXIT_SUCCESS;
}
