#pragma once

#include <vector>
#include <random>
#include <cmath> 
#include <array>

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
    vector<float> z1;
    vector<float> a1;
    vector<float> z2;
    vector<float> a2;
};

struct Gradients {
    vector<vector<float>> dW1;
    vector<float> db1;

    vector<vector<float>> dW2;
    vector<float> db2;
};

class NeuralNetwork {
private:
    Layer hidden;
    Layer output;
public:
    NeuralNetwork();

    vector<float> add_bias(const vector<float>& values, const vector<float>& biases);
    vector<float> apply_relu(const vector<float>& values);
    ForwardCache forward_prop(const vector<float>& X);
    vector<int> one_hot(const vector<float>& Y);
    Gradients backward_prop(const vector<float>& X, const ForwardCache& cache, int correct_label);
    void update_weights(Gradients& cache, const float alpha);
    void gradient_descent(const vector<Train_data>& train_data, const float alpha, 
        const int num_epochs, const vector<Train_data>& val_data);
    int predict(const vector<float>& X);
    float get_accuracy(const vector<Train_data>& dataset);
};


Layer create_layer(const int input_size, const int out_size);