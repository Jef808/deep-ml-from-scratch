#include "network/layer.h"
#include "network/initializer.h"
#include "network/activation.h"

#include "Eigen/Dense"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using namespace dmlfs;
using namespace Catch::Matchers;

TEST_CASE("Layer initialization produces correct dimensions", "[Layer]") {
  int inputSize = 5;
  int outputSize = 3;
  Layer layer(inputSize, outputSize, Initializer::Type::RANDOM);

  SECTION("Weights and biases dimensions are correct") {
    REQUIRE(layer.weights().rows() == outputSize);
    REQUIRE(layer.weights().cols() == inputSize);
    REQUIRE(layer.biases().rows() == outputSize);
    REQUIRE(layer.biases().cols() == 1);
  }
}

TEST_CASE("Xavier initializer produces weights with correct standard deviation", "[Layer]") {
  int inputSize = 10;
  int outputSize = 5;
  Layer layer(inputSize, outputSize, Initializer::Type::XAVIER);

  double stdDev = std::sqrt(2.0 / (inputSize + outputSize));
  Eigen::MatrixXd weights = layer.weights();

  SECTION("Weights have correct standard deviation") {
    double mean = weights.mean();
    double variance = (weights.array() - mean).square().sum() / (weights.rows() * weights.cols() - 1);
    double expected_variance = 2.0 / (inputSize + outputSize);
    REQUIRE_THAT(mean, WithinAbs(0.0, 0.15));
    REQUIRE_THAT(variance, WithinAbs(expected_variance, 0.1));
  }

  inputSize = 100;
  outputSize = 50;
  layer = Layer(inputSize, outputSize, Initializer::Type::XAVIER);

  stdDev = std::sqrt(2.0 / (inputSize + outputSize));
  weights = layer.weights();

  SECTION("Weights have correct standard deviation for larger dimensions") {
    double mean = weights.mean();
    double variance = (weights.array() - mean).square().sum() / (weights.rows() * weights.cols() - 1);
    double expected_variance = 2.0 / (inputSize + outputSize);
    REQUIRE_THAT(mean, WithinAbs(0.0, 0.015));
    REQUIRE_THAT(variance, WithinAbs(expected_variance, 0.001));
  }
}

TEST_CASE("Can Initialize layer with pre-defined weights and biases", "[Layer]") {
  Eigen::MatrixXd weights(3, 5);
  Eigen::MatrixXd biases(3, 1);
  weights << 1.0, 2.0, 3.0, 4.0, 5.0,
             6.0, 7.0, 8.0, 9.0, 10.0,
             11.0, 12.0, 13.0, 14.0, 15.0;
  biases << 1.0, 2.0, 3.0;

  Layer layer(weights, biases, Activation::Type::NONE);

  SECTION("Weights and biases are set correctly") {
    REQUIRE(layer.weights().isApprox(weights));
    REQUIRE(layer.biases().isApprox(biases));
  }
}

TEST_CASE("Layer forward propagation has correct dimension and basic properties", "[Layer]") {
  int inputSize = 5;
  int outputSize = 3;
  Layer layer(inputSize, outputSize, Initializer::Type::RANDOM);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(inputSize, 1);
  Eigen::MatrixXd output = layer.forward(input);

  SECTION("Output dimensions are correct") {
    REQUIRE(output.rows() == outputSize);
    REQUIRE(output.cols() == 1);
  }

  SECTION("Output is not all zeros") {
    REQUIRE_FALSE(output.isApprox(Eigen::MatrixXd::Zero(output.rows(), output.cols())));
  }

  SECTION("Output is not all ones") {
    REQUIRE_FALSE(output.isApprox(Eigen::MatrixXd::Ones(output.rows(), output.cols())));
  }

  SECTION("Output is not all negative ones") {
    REQUIRE_FALSE(output.isApprox(-Eigen::MatrixXd::Ones(output.rows(), output.cols())));
  }
}

TEST_CASE("Layer forward propagation with zero initializer produces zero output", "[Layer]") {
  int inputSize = 5;
  int outputSize = 3;
  Layer layer(inputSize, outputSize, Initializer::Type::ZERO);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(inputSize, 1);
  Eigen::MatrixXd output = layer.forward(input);

  REQUIRE(output.isApprox(Eigen::MatrixXd::Zero(outputSize, 1)));
}

TEST_CASE("Forward propagation with a layer with no activation function is equivalent to a linear transformation", "[Layer]") {
  int inputSize = 5;
  int outputSize = 3;
  Layer layer(inputSize, outputSize, Initializer::Type::RANDOM, Activation::Type::NONE);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(inputSize, 1);
  Eigen::MatrixXd output = layer.forward(input);

  Eigen::MatrixXd expectedOutput = layer.weights() * input + layer.biases();

  REQUIRE(output.isApprox(expectedOutput));
}

TEST_CASE("Backward propagation updates weights and biases correctly in simple case", "[Layer]") {
  int inputSize = 2;
  int outputSize = 1;
  Eigen::MatrixXd weights(2, 2);
  weights << 0.5, -0.5,
             1.0, 0.0;
  Eigen::MatrixXd biases(2, 1);
  biases << 0.1,
            -0.1;
  Layer layer(weights, biases, Activation::Type::RELU);

  Eigen::MatrixXd input(2, 1);
  input << 1.0,
           -1.0;
  Eigen::MatrixXd output = layer.forward(input);
  Eigen::MatrixXd expectedOutput(2, 1);
  expectedOutput << 1.1,
                    0.9;

  SECTION("Output of forward propagation is correct") {
    REQUIRE(output.isApprox(expectedOutput, 1e-9));
  }

  Eigen::MatrixXd dOutput(2, 1);
  dOutput << 0.5,
             -0.5;

  Eigen::MatrixXd dActivation = ReLU{}.derivative(output);
  Eigen::MatrixXd dZ = dOutput.array() * dActivation.array();

  // Since outputs are positive, derivative of ReLU is 1
  SECTION("Derivative of ReLU is correct") {
    REQUIRE(dActivation.isApprox(Eigen::MatrixXd::Ones(2, 1), 1e-9));
    REQUIRE(dZ.isApprox(dOutput, 1e-9));
  }

  Eigen::MatrixXd dW = dZ * input.transpose();
  Eigen::MatrixXd expectedDW(2, 2);
  expectedDW << 0.5, -0.5,
                -0.5, 0.5;
  Eigen::MatrixXd dB = dZ;
  Eigen::MatrixXd expectedDB(2, 1);
  expectedDB << 0.5,
                -0.5;

  Eigen::MatrixXd dInput = layer.backward(dOutput);

  Eigen::MatrixXd actualDW = layer.weights_grad();
  Eigen::MatrixXd actualDB = layer.biases_grad();

  SECTION("Derivative of weights and biases are correct") {
    REQUIRE(dW.isApprox(expectedDW, 1e-9));
    REQUIRE(actualDW.isApprox(expectedDW, 1e-9));

    REQUIRE(dB.isApprox(expectedDB, 1e-9));
    REQUIRE(actualDB.isApprox(expectedDB, 1e-6));
  }

  Eigen::MatrixXd expectedDInput = weights.transpose() * dZ;
  Eigen::MatrixXd manualDInput(2, 1);
  manualDInput << -0.25,
                  -0.25;

  SECTION("Derivative of input is correct") {
    REQUIRE(manualDInput.isApprox(expectedDInput, 1e-9));
    REQUIRE(dInput.isApprox(expectedDInput, 1e-9));
  }

  Eigen::MatrixXd expectedWeights(2, 2);
  expectedWeights << 0.45, -0.45,
                     1.05, -0.05;
  Eigen::MatrixXd expectedBias(2, 1);
  expectedBias << 0.05,
                  -0.05;

  layer.updateWeights(-0.1 * layer.weights_grad());
  layer.updateBiases(-0.1 * layer.biases_grad());

  SECTION("Weights and biases are updated correctly") {
    REQUIRE(layer.weights().isApprox(expectedWeights, 1e-9));
    REQUIRE(layer.weights().isApprox(weights - 0.1 * dW, 1e-9));
    REQUIRE(layer.biases().isApprox(expectedBias, 1e-9));
    REQUIRE(layer.biases().isApprox(biases - 0.1 * dB, 1e-9));
  }
}
