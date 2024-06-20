#include "optimizer.h"

namespace dmlfs {

SGD::SGD(double learningRate):
    m_learningRate{learningRate}
{
}

void SGD::update(Network& network) {
  for (auto& layer : network.layers()) {
    Network::Matrix weights_grad = layer->weights_grad();
    Network::Matrix biases_grad = layer->biases_grad();

    layer->updateWeights(-m_learningRate * weights_grad);
    layer->updateBiases(-m_learningRate * biases_grad);
  }
}

}  // namespace dmlfs
