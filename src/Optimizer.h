class Optimizer {
public:
    virtual void step(Layer& layer) = 0;
    virtual void backward(Layer& layer) = 0;
};

