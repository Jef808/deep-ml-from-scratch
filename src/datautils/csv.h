#ifndef CSV_H
#define CSV_H

#include <algorithm>
#include <concepts>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

namespace dmlfs {

template<typename T, typename Arg>
concept ConstructibleFrom = requires(Arg arg) {
  T{arg};
};

/**
 * @brief Read a CSV file and parse it into a vector of a specified type
 * @param filename Name of the file to read
 * @param parser Function to parse a vector of strings into the desired type
 * @return Vector of parsed data
 *
 * The parser function should take a vector of strings and return the desired type.
 * For example, with
 *
 * @code
 *   struct T {
 *     int id;
 *     double value;
 *     std::string name;
 *
 *     T(const std::vector<std::string>& input):
 *       id(std::stoi(input[0])),
 *       value(std::stod(input[1])),
 *       name(input[2])
 *     {
 *     }
 *   };
 * @endcode
 *
 * we can use `read_csv<T>(filename)` to read a CSV file directly as a vector of structs
 * of type `T`.
 */
template <typename T = std::vector<std::string>>
requires ConstructibleFrom<T, std::vector<std::string>>
std::vector<T> read_csv(const std::string& filename);

/**
 * @brief Read a CSV file and parse it into a vector of strings
 * @param filename Name of the file to read
 * @return Vector of parsed data
 *
 * This function is the base specialization of the general `read_csv` templated function.
 * It simply reads each fields and store them as strings.
 */
template <>
inline std::vector<std::vector<std::string>> read_csv<std::vector<std::string>>(const std::string& filename) {
  std::vector<std::vector<std::string>> result;
  std::string buffer;
  std::ifstream ifs{filename};
  while (std::getline(ifs, buffer)) {
    auto& line = result.emplace_back();
    std::istringstream iss{buffer};
    std::string token;
    while (std::getline(iss, token, ',')) {
      line.push_back(token);
    }
  }
  return result;
}

/**
 * @brief Read a CSV file and parse it into a vector of a specified type
 * @param filename Name of the file to read
 * @return Vector of parsed data
 *
 * This function is a specialization of the more general `read_csv` templated function.
 * It reads a CSV file and returns a vector of structs of type `T`.
 */
template <typename T>
requires ConstructibleFrom<T, std::vector<std::string>>
std::vector<T> read_csv(const std::string& filename) {
  std::vector<T> result;
  auto raw = read_csv(filename);

  std::transform(raw.begin()+1, raw.end(), std::back_inserter(result),
                 [](const auto& row) { return T{row}; });
  return result;
}

}  // namespace dmlfs


#endif /* CSV_H */
