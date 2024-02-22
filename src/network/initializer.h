#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "Eigen/Dense"

namespace dmlfs {

/**
 * @brief Abstract class for initializers
 */
struct Initializer {
  /**
   * @brief Pure virtual function to initialize the weights and biases
   */
  virtual void operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const = 0;

  /**
   * @brief Enum class to represent the type of initializer
   */
  enum class Type {
    ZERO,
    RANDOM,
    XAVIER
  };

  /**
   * @brief Initialize the weights and biases
   * @param type Type of initializer
   * @param weights Weights matrix
   * @param biases Biases matrix
   */
  static void apply(Initializer::Type initializerType, Eigen::MatrixXd& weights, Eigen::MatrixXd& biases);

  /**
   * @brief Virtual destructor
   */
  virtual ~Initializer() = default;
};

/**
 * @brief Concrete class for zero initializer
 */
struct ZeroInitializer: public Initializer {
  void operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const override;
};

/**
 * @brief Concrete class for random initializer
 */
struct RandomInitializer: public Initializer {
  void operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const override;
};

/**
 * @brief Concrete class for Xavier initializer
 */
struct XavierInitializer : public Initializer {
  void operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const override;
};

}  // namespace dmlfs


#endif /* INITIALIZER_H */
