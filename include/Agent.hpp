#ifndef AGENT_HPP
#define AGENT_HPP

#include "Models.hpp"

class Agent
{
public:
  Agent(int actionSize) : onlineNet(R2D2Agent(1, actionSize)), targetNet(R2D2Agent(1, actionSize))
  {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // create networks
    onlineNet.to(device);
    targetNet.to(device);
  }
  R2D2Agent onlineNet;
  R2D2Agent targetNet;
};

#endif // AGENT_HPP