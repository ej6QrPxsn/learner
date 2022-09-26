#ifndef AGENT_HPP
#define AGENT_HPP

#include "Models.hpp"

class Agent {
public:
  Agent(int actionSize)
      : onlineNet(R2D2Agent(1, actionSize)),
        targetNet(R2D2Agent(1, actionSize)) {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);
    onlineNet.to(device);

    // ターゲットモデルは勾配不要
    targetNet.detach_();
    targetNet.to(device);
  }
  R2D2Agent onlineNet;
  R2D2Agent targetNet;

  int trainCount = 0;
};

#endif // AGENT_HPP