#ifndef NEURALNET_HPP
#define NEURALNET_HPP
#include "clcontext.hpp"
#include <vector>
#include <cmath>

/*
struct layer {
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};
*/

// neural network for embedding training
class neuralNet {
private:
    int in;             // input dimension
    int out;            // output dimension
    int layers;         // number of hidden layers (excluding input and output)
    int type;           // type of activation funtion

    std::vector<std::vector<std::vector<float>>> weights;       // hidden weights (input + hidden layers + ouput)
    std::vector<std::vector<float>> biases;                     // biases for all weights layers
    std::vector<std::vector<float>> hLayers;                    // sums
    std::vector<std::vector<float>> activations;                // activations of sums
    std::vector<std::vector<std::vector<float>>> gweights;      // gradients of weights
    std::vector<std::vector<float>> gbiases;                    // gradients of biases (if needed)
    std::vector<std::vector<int>> layerDimensions;              // 2 rows (lenghth, breadth)

public:

#ifdef USE_OPENCL
    OpenCLContext& ocl;
    neuralNet(OpenCLContext& context, int in, int out, int layers);
#elif USE_CUDA || USE_CPU
    neuralNet() = default;
    neuralNet(int in, int out, int layers);
#endif

    void setIn(int in);
    void setOut(int out);
    void setLayers(int layers);
    void setType(int type);

    void getIn();
    void getOut();
    void getLayers();
    void getType();


    void initialisHe();
    void initialiseXavier();
    void initialiseLeCunn();


#ifdef USE_CPU 
    void forprop();
    void backprop();
#elif USE_CUDA
    void cuforprop();
    void cubackprop();
#elif USE_OPENCL
    void clforprop();
    void clbackprop();
#endif

    ~neuralNet() = default;
};

#endif