#ifndef MATRIX_REPRESENTATION_H
#define MATRIX_REPRESENTATION_H

#include "Eigen/Dense"

#include <numeric>
#include <unordered_set>
#include <vector>


namespace dmlfs {

template <size_t Nc, size_t Nd>
struct StructRepresentation {
  int id;
  std::array<double, Nc> continuous_features;
  std::array<int, Nd> categorical_features;
};


/**
 * @brief Count distinct categories for each categorical feature.
 * @param data Vector of structs
 * @return Vector of counts of distinct categories
 *
 * The input vector of structs should have a fixed number of continuous and categorical features.
 * The function returns a vector of counts of distinct categories for each categorical feature.
 */
template <size_t Nc, size_t Nd>
std::vector<int> count_distinct_categories(const std::vector<StructRepresentation<Nc, Nd>>& data) {
  std::vector<std::unordered_set<int>> category_sets(Nd);
  for (const auto& item : data) {
    for (int i = 0; i < Nd; ++i) {
      category_sets[i].insert(item.categorical_features[i]);
    }
  }
  std::vector<int> distinct_counts(Nd);
  for (const auto& set : category_sets) {
    distinct_counts.push_back(set.size());
  }
  return distinct_counts;
}

/**
 * @brief Convert a vector of structs to a matrix
 * @param data Vector of structs
 */
template <size_t Nc, size_t Nd>
Eigen::MatrixXd convert_to_matrix(const std::vector<StructRepresentation<Nc, Nd>>& data) {
  std::vector<int> distinct_categories = count_distinct_categories(data);
  int total_categories = std::accumulate(distinct_categories.begin(), distinct_categories.end(), 0);

  Eigen::MatrixXd _data(Nc + total_categories, data.size());

  for (size_t col = 0; col < data.sie(); ++col) {
    int row = 0;

    // Continuous features
    for (int c = 0; c < Nc; ++c) {
      _data(row, col) = data[col].continuous_features[c];
    }

    // One-hot encoding of categorical features
    for (int d = 0; d < Nd; ++d) {
      int category = data[col].categorical_features[d];
      int offset = std::accumulate(distinct_categories.begin(), distinct_categories.begin() + d, 0);
      for (int cat = 0; cat < distinct_categories[d]; ++cat) {
        _data(row + offset + cat, col) = (category == cat) ? 1.0 : 0.0;
      }
      row += distinct_categories[d];
    }
  }

  return _data;
}

}  // namespace dmlfs

#endif /* MATRIX_REPRESENTATION_H */
