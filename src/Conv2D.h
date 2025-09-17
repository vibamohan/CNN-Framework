class Conv2D : public Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd biases;
    Eigen::MatrixXd input_cache; // store for backward

public:
    Conv2D(int input_channels, int output_channels, int kernel_size);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void update_params(double learning_rate) override;
};

