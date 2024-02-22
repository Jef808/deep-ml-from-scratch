#include "Eigen/Dense"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using namespace Catch::Matchers;

TEST_CASE("Eigen and Catch2 Integration Test", "[integration]") {
  Eigen::Vector2d a(5.0, 6.0);
  Eigen::Vector2d b(2.0, 1.0);
  Eigen::Vector2d result = a + b;

  REQUIRE_THAT(result[0], WithinAbs(7.0, 0.000001));
  REQUIRE_THAT(result[1], WithinAbs(7.0, 0.000001));
}
