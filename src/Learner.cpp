#include "Learner.hpp"
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
std::array<AgentInput, NUM_TRAIN_THREADS> gAgentInputs;
std::array<TrainData, NUM_TRAIN_THREADS> gTrainDatas;

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

    action = inference(request, agentInput, device, localBuffer);

    size = send(fd_other, &action, sizeof(action), 0);
    if (size < 0) {
      perror("send");
    }

    steps++;

    if (gAgents[0].trainCount != prevTrainCount && steps % 100 == 0 &&
        request.envId == 0) {
      prevTrainCount = gAgents[0].trainCount;
      inferModel.copyFrom(gAgents[0].onlineNet);
    }
  }
}

int Learner::inference(Request &request,
                       AgentInput &agentInput, torch::Device device,
                       LocalBuffer &localBuffer) {
  int action;
  AgentOutput agentOutput;

  localBuffer.setInferenceParam(request, &agentInput);

  {
    c10::InferenceMode guard(true);
    agentOutput = inferModel.forward(agentInput);
  }

  auto q = agentOutput.q.cpu();

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
                                                agentOutput, policy);

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

  auto &agent = gAgents[threadNum];
  auto optimizer = new torch::optim::Adam(
      agent.onlineNet.parameters(),
      torch::optim::AdamOptions().lr(LEARNING_RATE).eps(EPSILON));

  auto &sampleData = gSamples[threadNum];
  auto &trainData = gTrainDatas[threadNum];
  AgentInput &input = gAgentInputs[threadNum];

  std::deque<float> lossList;

  while (1) {
    replay.sample(sampleData);
    dataConverter.toBatchedTrainData(trainData, sampleData.dataList);

    input.state = trainData.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)});
    input.prevAction =
        trainData.action.index({Slice(), Slice(0, REPLAY_PERIOD)});
    input.prevReward =
        trainData.reward.index({Slice(), Slice(0, REPLAY_PERIOD)});
    input.ih = &trainData.ih;
    input.hh = &trainData.hh;

    agent.onlineNet.forward(input);
    agent.targetNet.forward(input);

    input.state =
        trainData.state.index({Slice(), Slice(1 + REPLAY_PERIOD, None)});
    input.prevAction =
        trainData.action.index({Slice(), Slice(REPLAY_PERIOD, -1)});
    input.prevReward =
        trainData.reward.index({Slice(), Slice(REPLAY_PERIOD, -1)});
    input.ih = nullptr;
    input.hh = nullptr;

    auto onlineRet = agent.onlineNet.forward(input);
    auto targetRet = agent.targetNet.forward(input);

    auto [lossValue, priorities] = retraceLoss(
        trainData.action.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .unsqueeze(2),
        trainData.reward.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .squeeze(-1),
        trainData.done.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        trainData.policy.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        std::move(onlineRet.q), std::move(targetRet.q), device, optimizer);

    // 勾配を集計して設定
    updateGrad(threadNum, agent.onlineNet);

    // Update the parameters based on the calculated gradients.
    optimizer->step();

    agent.trainCount++;

    if (threadNum == 0) {
      lossList.push_back(lossValue);
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
    // std::cout << "stepsDone " << stepsDone << std::endl;
  }
}

int main(void) {
  int ret_code = 0;
  auto stateTensor =
      torch::zeros({1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
  int actionSize = 9;
  int numEnvs = NUM_ENVS;
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);

  Learner learner(stateTensor, actionSize, numEnvs, TRACE_LENGTH, REPLAY_PERIOD,
                  REPLAY_BUFFER_SIZE);

  // actorからのリクエスト受付
  auto inferThread = std::thread(&Learner::listenActor, &learner);

  inferThread.join();

  return EXIT_SUCCESS;
}
