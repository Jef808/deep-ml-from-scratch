#include "layer.h"

#include <cassert>

namespace dmlfs {

Layer::Layer(int inputSize, int outputSize, Initializer::Type initializerType, Activation::Type activationType):
    m_weights{Matrix::Zero(outputSize, inputSize)},
    m_weights_grad{Matrix::Zero(outputSize, inputSize)},
    m_biases{Matrix::Zero(outputSize, 1)},
    m_biases_grad{Matrix::Zero(outputSize, 1)},
    m_activation{nullptr}
{
  Initializer::apply(initializerType, m_weights, m_biases);
  Activation::set(activationType, m_activation);
}

Layer::Layer(const Matrix& weights, const Matrix& biases, Activation::Type activationType):
    m_weights{weights},
    m_weights_grad{Matrix::Zero(weights.rows(), weights.cols())},
    m_biases{biases},
    m_biases_grad{Matrix::Zero(biases.rows(), biases.cols())},
    m_activation{nullptr}
{
  Activation::set(activationType, m_activation);
}

Layer::Matrix Layer::forward(const Matrix& input) {
  assert(input.rows() == m_weights.cols());

  m_input = input;
  m_output = m_weights * input;
  m_output += m_biases.replicate(1, input.cols());

  return (*m_activation)(m_output);
}

Layer::Matrix Layer::backward(const Matrix& dOutput) {
  assert(dOutput.rows() == m_weights.rows());

  Matrix dActivation = m_activation->derivative(m_output);

  Matrix dZ = dOutput.array() * dActivation.array();
  m_weights_grad = dZ * m_input.transpose();
  m_biases_grad = dZ.rowwise().sum();

  return m_weights.transpose() * dZ;
}

void Layer::updateWeights(const Matrix& dWeights) {
  m_weights += dWeights;
}

void Layer::updateBiases(const Matrix& dBiases) {
  m_biases += dBiases;
}

}  // namespace dmlfs
