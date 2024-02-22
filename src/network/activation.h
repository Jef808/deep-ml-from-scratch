#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Eigen/Dense"

#include <memory>

namespace dmlfs {

/**
 * @brief Abstract class for activation functions
 */
struct Activation {
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) const = 0;
  virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) const = 0;

  /**
   * @brief Enum class to represent the type of activation function
   */
  enum class Type {
    NONE,
    RELU,
    SIGMOID,
    TANH
  };

  static void set(Activation::Type type, std::unique_ptr<Activation>& activation);

  virtual ~Activation() = default;
};

/**
 * @brief Concrete class for Trivial activation function
 */
struct Trivial: public Activation {
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) const override;
  Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) const override;
};

/**
 * @brief Concrete class for ReLU activation function
 */
struct ReLU: public Activation {
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) const override;
  Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) const override;
};

/**
 * @brief Concrete class for Sigmoid
 */
struct Sigmoid: public Activation {
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) const override;
  Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) const override;
};

/**
 * @brief Concrete class for Tanh
 */
struct Tanh: public Activation {
  Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) const override;
  Eigen::MatrixXd derivative(const Eigen::MatrixXd& input) const override;
};

}  // namespace dmlfs

#endif /* ACTIVATION_H */
