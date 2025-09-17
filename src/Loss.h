class Loss {
public:
    virtual double forward(const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) = 0;
    virtual Eigen::MatrixXd backward() = 0;
};

