#include <Eigen/Dense>

class Layer {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;

    virtual void update_params(double learning_rate) {}
    virtual ~Layer() = default;
};

