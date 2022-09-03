#ifndef DATA_CONVERTER_HPP
#define DATA_CONVERTER_HPP

#include <torch/torch.h>
#include "StructuredData.hpp"

using namespace torch::indexing;

class DataConverter {
public:
  DataConverter(torch::Tensor &state_, int actionSize_, int seqLength_)
      : state(state_), actionSize(actionSize_), seqLength(seqLength_) {}

  void toBatchedTrainData(TrainData &train, std::array<ReplayData, BATCH_SIZE> & dataList);

private:
  int actionSize;
  int seqLength;
  torch::Tensor state;
};

#endif // DATA_CONVERTER_HPP
