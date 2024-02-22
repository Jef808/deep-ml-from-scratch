#include "activation.h"

namespace dmlfs {

Eigen::MatrixXd Trivial::operator()(const Eigen::MatrixXd& input) const {
  return input;
}

Eigen::MatrixXd Trivial::derivative(const Eigen::MatrixXd& input) const {
  return Eigen::MatrixXd::Ones(input.rows(), input.cols());
}

Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd& input) const {
  return input.cwiseMax(0);
}

Eigen::MatrixXd ReLU::derivative(const Eigen::MatrixXd& input) const {
  return (input.array() > 0).cast<double>();
}

Eigen::MatrixXd Sigmoid::operator()(const Eigen::MatrixXd& input) const {
  return 1 / (1 + (-input.array()).exp());
}

Eigen::MatrixXd Sigmoid::derivative(const Eigen::MatrixXd& input) const {
  return operator()(input).array() * (1 - operator()(input).array());
}

Eigen::MatrixXd Tanh::operator()(const Eigen::MatrixXd& input) const {
  return input.array().tanh();
}

Eigen::MatrixXd Tanh::derivative(const Eigen::MatrixXd& input) const {
  return 1 - operator()(input).array().square();
}

void Activation::set(Activation::Type type, std::unique_ptr<Activation>& activation) {
  switch (type) {
    case Activation::Type::NONE:
      activation = std::make_unique<Trivial>();
      break;
    case Activation::Type::RELU:
      activation = std::make_unique<ReLU>();
      break;
    case Activation::Type::SIGMOID:
      activation = std::make_unique<Sigmoid>();
      break;
    case Activation::Type::TANH:
      activation = std::make_unique<Tanh>();
      break;
    default:
      throw std::invalid_argument("Unimplemented activation type");
  }
}

}  // namespace dmlfs
