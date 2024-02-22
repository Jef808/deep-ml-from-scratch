#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "network.h"

namespace dmlfs {

/**
 * @brief Abstract class for optimizers
 */
class Optimizer {
public:

  /**
   * @brief Update the weights and biases of the network
   * @param network Network to update
   */
  virtual void update(Network& network) = 0;

  /**
   * @brief Virtual destructor
   */
  virtual ~Optimizer() = default;
};

/**
 * @brief Concrete class for Stochastic Gradient Descent (SGD) optimizer
 */
class SGD : public Optimizer {
public:

  /**
   * @brief Constructor
   * @param learningRate Learning rate
   */
  explicit SGD(double learningRate);

  /**
   * @brief Update the weights and biases of the network according to the SGD algorithm
   * @param network Network to update
   */
  void update(Network& network) override;

private:

  /**
   * @brief Learning rate
   */
  double m_learningRate;
};

}  // namespace dmlfs


#endif // OPTIMIZER_H_
