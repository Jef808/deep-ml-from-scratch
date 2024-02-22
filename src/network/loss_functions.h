#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "Eigen/Dense"

namespace dmlfs {

/**
 * @brief Mean squared error loss function
 * @param y True labels
 * @param yHat Predicted labels
 * @return Mean squared error
 *
 * The mean squared error is defined as:
 * \f[
 *   L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
 * \f]
 * where \f$N\f$ is the number of samples, \f$y_i\f$ is the true label and \f$\hat{y}_i\f$ is the predicted label.
 * The mean squared error is a measure of the average of the squares of the errors or deviations.
 * It is a risk function corresponding to the expected value of the squared error loss.
 * The mean squared error is always non-negative and values closer to zero are better,
 * it is used in regression problems.
 * The mean squared error is also known as the quadratic loss, the quadratic mean, the quadratic score and the continuous ranked probability score.
 * it is the second moment of the error (about the origin) and is a measure of the quality of an estimator.
 *
 * @see meanSquaredErrorDerivative
 * @see https://en.wikipedia.org/wiki/Mean_squared_error
 */
double meanSquaredError(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

/**
 * @brief Derivative of the mean squared error loss function
 * @param y True labels
 * @param yHat Predicted labels
 * @return Derivative of the mean squared error
 *
 * The derivative of the mean squared error is defined as:
 * \f[
 *   \frac{\partial L(y, \hat{y})}{\partial \hat{y}} = \frac{\hat{y} - y}{N}
 * \f]
 * where \f$y\f$ is the true label and \f$\hat{y}\f$ is the predicted label.
 *
 * @see meanSquaredError
 */
Eigen::MatrixXd meanSquaredErrorDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

/**
 * @brief Cross entropy loss function
 * @param y True labels
 * @param yHat Predicted labels
 *
 * @return Cross entropy loss
 *
 * The cross entropy loss is defined as:
 *
 * \f[
 *   L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
 * \f]
 * where \f$N\f$ is the number of samples, \f$y_i\f$ is the true label and \f$\hat{y}_i\f$ is the predicted label.
 * The \f$\log\f$ function is clipped to avoid numerical instability.
 *
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 */
double crossEntropy(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

/**
 * @brief Derivative of the cross entropy loss function
 * @param y True labels
 * @param yHat Predicted labels
 * @return Derivative of the cross entropy loss
 *
 * The derivative of the cross entropy loss is defined as:
 * \f[
 *   \frac{\partial L(y, \hat{y})}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y} (1 - \hat{y})}
 * \f]
 * where \f$y\f$ is the true label and \f$\hat{y}\f$ is the predicted label.
 * The \f$\log\f$ function is also clipped to avoid numerical instability.
 *
 * @see crossEntropy
 */
Eigen::MatrixXd crossEntropyDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

/**
 * @brief Binary cross entropy loss function
 * @param y True labels
 * @param yHat Predicted labels
 * @return Binary cross entropy loss
 *
 * The binary cross entropy loss is defined as:
 * \f[
 *  L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
 * \f]
 * where \f$N\f$ is the number of samples, \f$y_i\f$ is the true label and \f$\hat{y}_i\f$ is the predicted label.
 * It is equivalent to the cross entropy loss function when the number of classes is 2.
 *
 * @see crossEntropy
 */
double binaryCrossEntropy(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

/**
 * @brief Derivative of the binary cross entropy loss function
 * @param y True labels
 * @param yHat Predicted labels
 * @return Derivative of the binary cross entropy loss
 *
 * The derivative of the binary cross entropy loss is defined as:
 * \f[
 *   \frac{\partial L(y, \hat{y})}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y} (1 - \hat{y})}
 * \f]
 * where \f$y\f$ is the true label and \f$\hat{y}\f$ is the predicted label.
 *
 * @see crossEntropy
 */
Eigen::MatrixXd binaryCrossEntropyDerivative(const Eigen::MatrixXd& y, const Eigen::MatrixXd& yHat);

}  // namespace dmlfs


#endif /* LOSS_FUNCTIONS_H */
