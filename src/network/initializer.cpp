#include "initializer.h"

#include <random>

namespace dmlfs {

void ZeroInitializer::operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const {
  weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
  biases = Eigen::MatrixXd::Zero(biases.rows(), 1);
}

void RandomInitializer::operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  for (int i=0; i<weights.rows(); ++i) {
    for (int j=0; j<weights.cols(); ++j) {
      weights(i, j) = dis(gen);
    }
  }
  biases = Eigen::MatrixXd::Random(biases.rows(), 1) * 0.01;
}

void XavierInitializer::operator()(Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) const {
  double stdDev = std::sqrt(2.0 / (weights.rows() + weights.cols()));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dis(0.0, stdDev);
  for (int i=0; i<weights.rows(); ++i) {
    for (int j=0; j<weights.cols(); ++j) {
      weights(i, j) = dis(gen);
    }
  }
  biases = Eigen::MatrixXd::Random(biases.rows(), 1) * 0.01;
}

void Initializer::apply(Initializer::Type initializerType, Eigen::MatrixXd& weights, Eigen::MatrixXd& biases) {
  switch (initializerType) {
    case Initializer::Type::ZERO:
      break;
    case Initializer::Type::RANDOM:
      RandomInitializer()(weights, biases);
      break;
    case Initializer::Type::XAVIER:
      XavierInitializer()(weights, biases);
      break;
    default:
      throw std::invalid_argument("Unimplemented initializer type");
  }
}

}  // namespace dmlfs
