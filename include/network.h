#pragma once

#include <vector>
#include <random>
#include <cmath> 
#include <array>
#include <stdexcept>

#include "./math_functions.h"
#include "./ml_functions.h"
#include "./mnist.h"

using std::vector;
using std::random_device;
using std::mt19937;
using std::normal_distribution;
using std::invalid_argument;
using std::max;
using std::array;

struct Layer{
    vector<vector<float>> weights;
    vector<float> biases;
};

struct ForwardCache {
    // vector<float> z1;
    // vector<float> a1;
    // vector<float> z2;
    // vector<float> a2;

    vector<vector<float>> z;
    vector<vector<float>> a;
};

struct Gradients {
    // vector<vector<float>> dW1;
    // vector<float> db1;

    // vector<vector<float>> dW2;
    // vector<float> db2;

    vector<vector<vector<float>>> dW;
    vector<vector<float>> db;
};

class NeuralNetwork {
private:
    // Layer hidden;
    // Layer output;

    vector<Layer> layers;
    vector<int> layer_sizes;
public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(const vector<int>& layer_sizes_);
    void add_layer(int output_size);
    void add_layer(int input_size, int output_size);


    size_t num_layers() const {return layers.size();}
    const vector<int>& get_layer_sizes() const { return layer_sizes; }

    ForwardCache forward_prop(const vector<float>& X);
    Gradients backward_prop(const vector<float>& X, const ForwardCache& cache, int correct_label);
    void update_weights(Gradients& cache, const float alpha);
    void gradient_descent(const vector<Train_data>& train_data, const float alpha, 
        const int num_epochs, const vector<Train_data>& val_data);
    int predict(const vector<float>& X);
    float get_accuracy(const vector<Train_data>& dataset);


private:
    vector<float> add_bias(const vector<float>& values, const vector<float>& biases);
    vector<float> apply_relu(const vector<float>& values);
    vector<int> one_hot(const vector<float>& Y);
};


Layer create_layer(const int input_size, const int out_size);