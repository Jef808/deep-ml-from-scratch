#include "optimizer.h"

namespace dmlfs {

SGD::SGD(double learningRate):
    m_learningRate{learningRate}
{
}

void SGD::update(Network& network) {
  for (auto& layer : network.layers()) {
    Network::Matrix weightsGradient = layer->weightsGradients();
    Network::Matrix biasesGradient = layer->biasesGradients();

    layer->updateWeights(-m_learningRate * weightsGradient);
    layer->updateBiases(-m_learningRate * biasesGradient);
  }
}

}  // namespace dmlfs
