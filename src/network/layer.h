#ifndef LAYER_H
#define LAYER_H

#include "activation.h"
#include "initializer.h"
#include "CommonMacros.h"

#include "Eigen/Dense"

#include <memory>

namespace dmlfs {

/**
 * @brief Class representing a layer in a neural network
 */
class Layer {
public:
  using Matrix = Eigen::MatrixXd;
  
  /**
   * @brief Constructor
   * @param inputSize Number of input neurons
   * @param outputSize Number of output neurons
   * @param initializerType Type of initializer
   * @param activationType Type of activation function
   */
  Layer(int inputSize,
        int outputSize,
        Initializer::Type initializerType = Initializer::Type::ZERO,
        Activation::Type activationType = Activation::Type::NONE);

  /**
   * @brief Constructor specifying weights and biases
   * @param weights Weights matrix
   * @param biases Biases matrix
   * @param activationType Type of activation function
   */
  Layer(const Matrix& weights,
        const Matrix& biases,
        Activation::Type activationType = Activation::Type::NONE);

  /**
   * @brief Getter for the weights matrix
   *
   * @see CommonMacros.h
   */
  DEFINE_CONST_GETTER(Matrix, weights);

  /**
   * @brief Getter for the weights gradients matrix
   *
   * @see CommonMacros.h
   */
  DEFINE_CONST_GETTER(Matrix, weightsGradients);

  /**
   * @brief Getter for the biases matrix
   *
   * @see CommonMacros.h
   */
  DEFINE_CONST_GETTER(Matrix, biases);

  /**
   * @brief Getter for the biases gradients matrix
   *
   * @see CommonMacros.h
   */
  DEFINE_CONST_GETTER(Matrix, biasesGradients);

  /**
   * @brief Forward propagation
   * @param input Input to the layer
   * @return Output of the layer
   */
  Matrix forward(const Eigen::MatrixXd& input);

  /**
   * @brief Backward propagation
   * @param dOutput Derivative of the output
   * @param learningRate Learning rate
   * @return Derivative of the input
   */
  Matrix backward(const Matrix& dOutput);

  /**
   * @brief Update weights after forward and backward propagation
   * @param dWeights Gradients with which to update the weights
   */
  void updateWeights(const Matrix& dWeights);

  /**
   * @brief Update biases after forward and backward propagation
   * @param dBiases Gradients with which to update the biases
   */
  void updateBiases(const Matrix& dBiases);

private:
  /**
   * @brief Weights
   */
  Matrix m_weights;

  /**
   * @brief Weights gradients
   */
  Matrix m_weightsGradients;

  /**
   * @brief Biases
   */
  Matrix m_biases;

  /**
   * @brief Biases gradients
   */
  Matrix m_biasesGradients;

  /**
   * @brief The input to the layer for use in backpropagation
   */
  Matrix m_input;

  /**
   * @brief The pre-activation vectors for use in backpropagation
   */
  Matrix m_output;

  /**
   * @brief Activation function and its derivative
   */
  std::unique_ptr<Activation> m_activation;
};

}  // namespace dmlfs

#endif // LAYER_H
