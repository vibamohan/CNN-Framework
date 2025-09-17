#include "Dense.h"
#include <random>
#include <iostream>

Dense::Dense(int input_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.01); // might need to tune this -- theoretically it should tune to a good value anyways though

    weights = Eigen::MatrixXd(output_size, input_size).unaryExpr([&](double) { return d(gen); });
    biases = Eigen::VectorXd::Zero(output_size);
}

Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    return (weights * input).colwise() + biases;
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& grad_output) {
    grad_weights = grad_output * input_cache.transpose();
    grad_biases = grad_output.rowwise().sum();
    Eigen::MatrixXd grad_input = weights.transpose() * grad_output;
    return grad_input;
}

void Dense::update_params(double learning_rate) {
    weights -= learning_rate * grad_weights;
    biases  -= learning_rate * grad_biases;
}

