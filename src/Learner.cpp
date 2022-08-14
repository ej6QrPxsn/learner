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
std::mutex mtx_;

const auto STATE =
    torch::empty({1, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));

std::array<std::vector<ReplayData>, NUM_ENVS> gReplayDatas;
std::array<std::vector<RetraceQ>, NUM_ENVS> gRetraceQs;
std::vector<AgentInput> gInferInputs(NUM_ENVS, AgentInput(STATE, 1, 1));
// std::vector<AgentOutput> gInferOutputs(NUM_ENVS, AgentOutput(1, 1));
std::vector<RetraceData> gRetraceDatas(NUM_ENVS, RetraceData(BATCH_SIZE * 2,
                                                             1 + TRACE_LENGTH,
                                                             ACTION_SIZE));

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
  int steps = 0;

  // 初回のみパケットサイズを得る
  size = recv(fd_other, &buf_len, sizeof(buf_len), 0);
  // printf("buf_len = %d\n", buf_len);

  while (1) {
    // データ本体の受信
    size = recv(fd_other, buf, buf_len, 0);
    // printf("buf_len = %d\n", buf_len);

    int task = *reinterpret_cast<int *>(&buf[0]);

    request.state = torch::from_blob(&buf[4], state.sizes(), state.dtype());
    request.reward = *reinterpret_cast<float *>(&buf[buf_len - 5]);
    request.done = *reinterpret_cast<bool *>(&buf[buf_len - 1]);

    action = inference(task, request);

    size = send(fd_other, &action, sizeof(action), 0);
    if (size < 0) {
      perror("send");
    }

    steps++;

   // アクターネットワーク切り替え
    if (task == 0 && steps % ACTOR_UPDATE == 0) {
      inferModelManager.switchUseModel();
    }
  }
}

int Learner::inference(int envId, Request &request) {
  int action;
  std::vector<ReplayData> &replayDatas = gReplayDatas[envId];
  std::vector<RetraceQ> &retraceQs = gRetraceQs[envId];

  AgentInput &inferInput = gInferInputs[envId];
  AgentOutput inferOutput(1, 1);

  localBuffer.setInferenceParam(envId, request, &inferInput);

  {
    c10::InferenceMode guard(true);
    inferModelManager.getModel().forward(inferInput, &inferOutput, true);
  }

  // 選択アクションの確率
  auto policy = torch::amax(torch::softmax(inferOutput.q, 2), 2);

  // std::cout << "policies: " << policies << std::endl;

  // 環境ごとの閾値
  auto epsThreshold =
      torch::pow(0.4, torch::linspace(1., 8., numEnvs)).index({envId});

  auto randomAction =
      torch::randint(0, inferOutput.q.size(2), 1).to(torch::kLong);
  auto prob = torch::randn(1);

  auto selectAction =
      torch::where(prob < epsThreshold, randomAction, torch::argmax(inferOutput.q, 2));

  localBuffer.updateAndGetTransition(envId, request, selectAction, inferOutput,
                                     policy, &replayDatas, &retraceQs);

  if (!replayDatas.empty()) {

    RetraceData &retraceData = gRetraceDatas[envId];

    int returnSize = replayDatas.size();
    dataConverter.toBatchedRetraceData(replayDatas, retraceQs, &retraceData,
                                       returnSize);

    auto priorities = std::get<1>(
        retraceLoss(retraceData.action.index({Slice(None, returnSize)}),
                    retraceData.reward.index({Slice(None, returnSize)}),
                    retraceData.done.index({Slice(None, returnSize)}),
                    retraceData.policy.index({Slice(None, returnSize)}),
                    retraceData.onlineQ.index({Slice(None, returnSize)}),
                    retraceData.targetQ.index({Slice(None, returnSize)})));

    replay.putReplayQueue(priorities, replayDatas);
    replayDatas.clear();
    retraceQs.clear();
  }

  return selectAction.item<int>();
}

void Learner::trainLoop() {
  int stepsDone = 0;
  // std::string envent_file = "/home/yoshi/logs";
  // tensorflow::EventsWriter writer(envent_file);

  auto optimizer = torch::optim::Adam(
      agent.onlineNet.parameters(),
      torch::optim::AdamOptions().lr(LEARNING_RATE).eps(EPSILON));

  std::deque<float> lossList;

  AgentOutput output(BATCH_SIZE, REPLAY_PERIOD);
  AgentOutput onlineOutput(BATCH_SIZE, TRACE_LENGTH);
  AgentOutput targeOutput(BATCH_SIZE, TRACE_LENGTH);

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

    AgentInput input;

    input.state = data.state.index({Slice(), Slice(1, 1 + REPLAY_PERIOD)});
    input.prevAction = data.action.index({Slice(), Slice(0, REPLAY_PERIOD)});
    input.prevReward = data.reward.index({Slice(), Slice(0, REPLAY_PERIOD)});
    input.ih = &data.ih;
    input.hh = &data.hh;

    agent.onlineNet.forward(input, &output);
    agent.targetNet.forward(input, &output);

    input.state = data.state.index({Slice(), Slice(1 + REPLAY_PERIOD, None)});
    input.prevAction = data.action.index({Slice(), Slice(REPLAY_PERIOD, -1)});
    input.prevReward = data.reward.index({Slice(), Slice(REPLAY_PERIOD, -1)});
    input.ih = nullptr;
    input.hh = nullptr;

    agent.onlineNet.forward(input, &onlineOutput);
    agent.targetNet.forward(input, &targeOutput);

    auto retraceRet = retraceLoss(
        data.action.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .unsqueeze(2),
        data.reward.index({Slice(), Slice(1 + REPLAY_PERIOD, None)})
            .squeeze(-1),
        data.done.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        data.policy.index({Slice(), Slice(1 + REPLAY_PERIOD, None)}),
        onlineOutput.q, targeOutput.q);

    auto losses = std::get<0>(retraceRet);
    auto priorities = std::get<1>(retraceRet);

    auto loss = torch::mean(losses);
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

    // 推論ネットワークのパラメータ更新
    inferModelManager.setModelParams(agent.onlineNet);

    // ターゲットネットワーク更新
    if (stepsDone % TARGET_UPDATE == 0) {
      agent.targetNet.copyFrom(agent.onlineNet);
    }

    // tensorflow::Event event;
    // tensorflow::Summary::Value* summ_val =
    // event.mutable_summary()->add_value(); event.set_step(stepsDone);
    // summ_val->set_tag("loss");
    // summ_val->set_simple_value(loss);
    // writer->WriteEvent(event);

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
  int numEnvs = NUM_ENVS;

  Learner learner(stateTensor, actionSize, numEnvs, TRACE_LENGTH, REPLAY_PERIOD,
                  REPLAY_BUFFER_SIZE);

  // actorからのリクエスト受付
  auto inferLoop = std::thread(&Learner::listenActor, &learner);

  // optimizer = optim.Adam(agent.online_net.parameters(), lr=LEARNING_RATE,
  // eps=1e-3)

  learner.trainLoop();

  inferLoop.join();

  return EXIT_SUCCESS;
}
