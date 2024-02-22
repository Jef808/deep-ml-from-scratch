#include "network.h"

namespace dmlfs {

Network& Network::addLayer(std::shared_ptr<Layer> layer) {
  m_layers.push_back(layer);
  return *this;
}

Network::Matrix Network::forward(const Matrix& input) {
  Matrix output = input;
  for (auto& layer : m_layers) {
    output = layer->forward(output);
  }
  return output;
}

void Network::backward(const Matrix& dLoss_Output) {
  Matrix dOutput = dLoss_Output;
  for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
    dOutput = (*it)->backward(dOutput);
  }
}

}  // namespace dmlfs
