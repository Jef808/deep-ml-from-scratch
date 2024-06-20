#include "network/network.h"
#include "datautils/csv.h"

#include "catch2/catch_test_macros.hpp"

#include "Eigen/Dense"

#include <algorithm>
#include <numeric>
#include <iostream>


namespace dmlfs {

TEST_CASE("Network can be trained as a simple binary classifier", "[network]") {
    // Create a simple dataset
    Eigen::MatrixXd X(4, 2);
    X << 0, 0,
         0, 1,
         1, 0,
         1, 1;

    Eigen::MatrixXd y(4, 1);
    y << 0, 1, 1, 0;

    // Create a simple network
    Network network;
    network.add_layer(2, 2, Activation::sigmoid);
    network.add_layer(2, 1, Activation::sigmoid);

    // Train the network
    network.train(X, y, 1000, 0.1);

    // Check the predictions
    Eigen::MatrixXd y_pred = network.predict(X);
    REQUIRE(y_pred.rows() == y.rows());
    REQUIRE(y_pred.cols() == y.cols());

    for (int i = 0; i < y.rows(); ++i) {
        REQUIRE(y_pred(i, 0) == Approx(y(i, 0)).margin(0.1));
    }

} // namespace dmlfs
