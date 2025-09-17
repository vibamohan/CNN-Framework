#include "Eigen/MatrixXd"

class ActivationLayer : public Layer {
private:
    Eigen::MatrixXd mask; // for backward
    Eigen::MatrixXd input;

public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
};

