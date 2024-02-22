#include "loss_functions.h"

namespace dmlfs {

double meanSquaredError(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  Eigen::MatrixXd diff = yHat - y;
  return (diff.array().square()).mean();
}

Eigen::MatrixXd meanSquaredErrorDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  Eigen::MatrixXd gradient = -(y - yHat);
  return gradient;
}

double crossEntropy(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  const double epsilon = 1e-15;
  Eigen::MatrixXd yHatClipped = yHat.unaryExpr([epsilon](double x) { return std::max(epsilon, std::min(1 - epsilon, x)); });

  Eigen::ArrayXXd lossArray = -(y.array() * yHatClipped.array().log()).colwise().sum();
  return lossArray.mean();
}

Eigen::MatrixXd crossEntropyDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  const double epsilon = 1e-15;
  Eigen::MatrixXd yHatClipped = yHat.unaryExpr([epsilon](double x) { return std::max(epsilon, std::min(1 - epsilon, x)); });

  Eigen::MatrixXd gradient = -(y.array() / yHatClipped.array());
  return gradient;
}

double binaryCrossEntropy(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  const double epsilon = 1e-15;
  Eigen::MatrixXd yHatClipped = yHat.unaryExpr([epsilon](double x) { return std::max(epsilon, std::min(1 - epsilon, x)); });

  Eigen::ArrayXXd lossArray = -y.array() * yHatClipped.array().log() - (1 - y.array()) * (1 - yHatClipped.array()).log();
  return lossArray.sum() / y.rows();
}

Eigen::MatrixXd binaryCrossEntropyDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat) {
  const double epsilon = 1e-15;
  Eigen::MatrixXd yHatClipped = yHat.unaryExpr([epsilon](double x) { return std::max(epsilon, std::min(1 - epsilon, x)); });

  Eigen::ArrayXXd gradient = - (y.array() / yHatClipped.array()) + (1 - y.array()) / (1 - yHatClipped.array());
  return gradient.matrix();
}

}  // namespace dmlfs
