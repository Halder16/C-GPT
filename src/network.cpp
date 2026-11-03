#include "../include/network.h"


//helper,
vector<float> column_sum(const vector<vector<float>>& mat) {

    int rows = mat.size();
    int cols = mat[0].size();

    vector<float> result(rows, 0.0f);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += mat[i][j];
        }
    }

    return result;
}


Layer create_layer(const int input_size, const int out_size){
    Layer layer;

    layer.weights.resize(out_size, vector<float> (input_size));
    layer.biases.resize(out_size, 0.0f);

    static random_device rd;
    static mt19937 gen(rd());

    float std_dev = sqrt(2.0f / input_size);
    normal_distribution<float> dist(0.0f, std_dev);

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer.weights[i][j] = dist(gen);
        }
    }

    return layer;
}



NeuralNetwork::NeuralNetwork(){
    hidden = create_layer(784, 128);
    output = create_layer(128, 10);
}


vector<float> NeuralNetwork::add_bias(const vector<float>& values, const vector<float>& biases){
    return vector_add(values, biases);
}


vector<float> NeuralNetwork::apply_relu(const vector<float>& values){
    vector<float> results(values.size());

    for(size_t i=0;i<values.size();i++){
        results[i] = relu(values[i]);
    }

    return results;
}


ForwardCache NeuralNetwork::forward_prop(const vector<float>& X){
    //
    if (X.size() != 784) {
        throw invalid_argument("Expected 784 input values");
    }

    //hidden layer,
    vector<float> Z1 = mat_vec_mult(hidden.weights, X);
    Z1 = add_bias(Z1, hidden.biases);

    vector<float> A1 = apply_relu(Z1);

    //output layer,
    vector<float> Z2 = mat_vec_mult(output.weights, A1);
    Z2 = add_bias(Z2, output.biases);

    //predictions,
    vector<float> A2 = softmax(Z2);

    return {Z1, A1, Z2, A2};
}


vector<int> NeuralNetwork::one_hot(const vector<float>& Y){
    // this is applied on the usser supplied true labels, Y
    if (Y.empty()) {
        throw invalid_argument("Cannot one-hot encode empty predictions");
    }

    vector<int> results(Y.size(), 0);

    float max_ele = -5.0f; //shouldnt be smaller than 0 anyway, right?
    int idx = -1;

    for(size_t i=0;i<Y.size();i++){
        if(max_ele < Y[i]){
            max_ele = Y[i];
            idx = i;
        }
    }

    results[idx] = 1;
    return results;
}


Gradients NeuralNetwork::backward_prop(const vector<float>& X, const ForwardCache& cache, int correct_label){
    if (X.size() != hidden.weights[0].size()) {
        throw invalid_argument("Expected 784 input values");
    }
    if (cache.z1.size() != hidden.weights.size() ||
        cache.a1.size() != hidden.weights.size() ||
        cache.z2.size() != output.weights.size() ||
        cache.a2.size() != output.weights.size()) {
        throw invalid_argument("Forward cache dimensions do not match the network");
    }
    if (correct_label < 0 || correct_label >= static_cast<int>(cache.a2.size())) {
        throw invalid_argument("Correct label must be between 0 and 9");
    }

    Gradients grads;

    // Softmax with cross-entropy: dZ2 = A2 - one_hot(correct_label).
    vector<float> dZ2 = cache.a2;
    dZ2[correct_label] -= 1.0f;

    // One image means m = 1, so no averaging is needed.
    grads.dW2.resize(dZ2.size(), vector<float>(cache.a1.size()));
    for (size_t i = 0; i < dZ2.size(); ++i) {
        for (size_t j = 0; j < cache.a1.size(); ++j) {
            grads.dW2[i][j] = dZ2[i] * cache.a1[j];
        }
    }
    grads.db2 = dZ2;

    vector<float> dZ1 = mat_vec_mult(mat_transpose(output.weights), dZ2);
    for (size_t i = 0; i < dZ1.size(); ++i) {
        dZ1[i] *= relu_derivative(cache.z1[i]);
    }

    grads.dW1.resize(dZ1.size(), vector<float>(X.size()));
    for (size_t i = 0; i < dZ1.size(); ++i) {
        for (size_t j = 0; j < X.size(); ++j) {
            grads.dW1[i][j] = dZ1[i] * X[j];
        }
    }
    grads.db1 = dZ1;

    return grads;
}
