#include "network.h"
#include "datautils/csv.h"

#include "catch2/catch_test_macros.hpp"

#include "Eigen/Dense"

#include <array>
#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <tuple>

namespace dmlfs {

struct T {
  enum Class {
    IrisSetosa = 1,
    IrisVersicolor = 2,
    IrisVirginica = 3
  };
  std::array<double, 4> features;
  Class species;

  T(const std::vector<std::string>& input):
    features{
      std::stod(input[1]),
      std::stod(input[2]),
      std::stod(input[3]),
      std::stod(input[4])},
    species([sz=input[5].size()](){
      switch (sz) {
        case 11: return Class::IrisSetosa;
        case 15: return Class::IrisVersicolor;
        case 14: return Class::IrisVirginica;
        default:
          throw;
      }
    }())
  {
  }
};


TEST_CASE("Read CSV", "[Iris]") {
  auto _data = read_csv<T>("/home/jfa/projects/dml-from-scratch/data/iris/Iris.csv");

  for (const auto& row : _data) {
    std::cout << row.features[0] << " " << row.features[1] << " " << row.features[2] << " " << row.features[3] << " " << row.species << std::endl;
  }

  REQUIRE(_data.size() == 150);

  // Select two features for binary classification:
  // Iris-setosa and Iris-versicolor
  auto data = std::vector<T>{_data.begin(), _data.begin() + 101};

}

}  // namespace dmlfs
