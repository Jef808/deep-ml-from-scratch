#ifndef NETWORK_H
#define NETWORK_H

#include "CommonMacros.h"
#include "layer.h"

#include <memory>
#include <vector>

namespace dmlfs {

class Network {
public:
  using Matrix = Layer::Matrix;

  /**
   * @brief Default constructor
   */
  Network() = default;

  /**
   * @brief Add a layer to the network
   * @param layer Layer to add
   * @return Reference to this network
   */
  Network& addLayer(std::shared_ptr<Layer> layer);

  /**
   * @brief Forward pass through the network
   * @param input Input matrix
   * @return Output matrix
   */
  Matrix forward(const Matrix& input);

  /**
   * @brief Backward pass through the network
   * @param dLoss_Output Gradient of the loss with respect to the output
   */
  void backward(const Matrix& dLoss_Output);

  /**
   * @brief Read-write getter for the layers
   *
   * @see CommonMacros.h
   */
  DEFINE_GETTER(std::vector<std::shared_ptr<Layer>>, layers);

private:

  /**
   * @brief Layers in the network
   */
  std::vector<std::shared_ptr<Layer>> m_layers;
};

} // namespace dmlfs


#endif /* NETWORK_H */
