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


void NeuralNetwork::add_layer(int output_size){
    if (layers.empty()) {
        throw invalid_argument("First layer needs explicit input_size. Use add_layer(input_size, output_size)");
    }
    int input_size = layer_sizes.back();  // Previous layer's output
    layers.push_back(create_layer(input_size, output_size));
    layer_sizes.push_back(output_size);
}



void NeuralNetwork::add_layer(int input_size, int output_size) {
    if (!layers.empty() && input_size != layer_sizes.back()) {
        throw invalid_argument("Input size must match previous layer's output size");
    }
    layers.push_back(create_layer(input_size, output_size));
    layer_sizes.push_back(output_size);
}



NeuralNetwork::NeuralNetwork(const vector<int>& layer_sizes_) {
    if (layer_sizes_.size() < 2) {
        throw invalid_argument("Need at least input and output sizes");
    }
    layer_sizes = layer_sizes_;
    
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        layers.push_back(create_layer(layer_sizes[i-1], layer_sizes[i]));
    }
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
    if (X.size() != layers[0].weights[0].size()) {  // First layer's input size
        throw invalid_argument("Input size mismatch");
    }

    ForwardCache cache;
    cache.a.push_back(X); //a[0]


    for(size_t i=0;i<layers.size();i++){
        const Layer& layer = layers[i];

        // Linear z = W*a_prev + b
        vector<float> z = mat_vec_mult(layer.weights, cache.a.back());
        z = add_bias(z, layer.biases);
        cache.z.push_back(z);

        //activation
        vector<float> a;
        if(i==layers.size()-1){ //output layer
            a = softmax(z);
        }else{
            a = apply_relu(z); //hidden layer
        }
        cache.a.push_back(a);
    }

    return cache;

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
    Gradients grads;
    grads.dW.resize(layers.size());
    grads.db.resize(layers.size());

    int num_layers = layers.size();
    int num_classes = layers.back().weights.size(); //output layer size

    vector<float> dZ = cache.a.back(); // a[L] = output activations
    dZ[correct_label] -= 1.0f; // dL/dZ = A - Y_one_hot

    //backprop in reverse
    for(int i=num_layers-1;i>=0;i--){
        const vector<float>& a_prev = cache.a[i]; //input to this layer

        //dW = dZ*a_prev^T
        grads.dW[i].resize(dZ.size(), vector<float>(a_prev.size()));
        for(size_t row=0;row<dZ.size();row++){
            for(size_t col=0;col<a_prev.size();col++){
                grads.dW[i][row][col] = dZ[row]*a_prev[col];
            }
        }
        grads.db[i] = dZ; //since bias added directly

        if(i>0){
            //dZ_prev = W-curr^T * dZ * relu'(z_prev)
            vector<float> dZ_prev = mat_vec_mult(mat_transpose(layers[i].weights), dZ);
            for(size_t j=0;j<dZ_prev.size();j++){
                dZ_prev[j] *= relu_derivative(cache.z[i-1][j]);
            }
            dZ = std::move(dZ_prev);
        }
    }
    return grads;
}




void NeuralNetwork::update_weights(Gradients& grads, const float alpha){
    // W1 = W1 - alpha * dW1
    // b1 = b1 - alpha * db1    
    // W2 = W2 - alpha * dW2  
    // b2 = b2 - alpha * db2 

    for (size_t i = 0; i < layers.size(); ++i) {
        // W = W - alpha * dW
        for (size_t r = 0; r < layers[i].weights.size(); ++r) {
            for (size_t c = 0; c < layers[i].weights[r].size(); ++c) {
                layers[i].weights[r][c] -= alpha * grads.dW[i][r][c];
            }
        }
        // b = b - alpha * db
        for (size_t j = 0; j < layers[i].biases.size(); ++j) {
            layers[i].biases[j] -= alpha * grads.db[i][j];
        }
    }
}



int NeuralNetwork::predict(const vector<float>& X){
    ForwardCache cache = forward_prop(X);
    return argmax(cache.a.back());
}



float NeuralNetwork::get_accuracy(const vector<Train_data>& dataset){
    int correct = 0;
    for (const auto& sample : dataset) {
        if (predict(sample.image) == sample.label) correct++;
    }
    return static_cast<float>(correct) / dataset.size();
}



void NeuralNetwork::gradient_descent(const vector<Train_data>& train_data, 
    const float alpha, const int num_epochs, const vector<Train_data>& val_data){

    for(int epoch=0;epoch<num_epochs;epoch++){
        for(const auto& sample: train_data){
            ForwardCache fcache = forward_prop(sample.image);
            Gradients grads = backward_prop(sample.image, fcache, sample.label);
            update_weights(grads, alpha);
        }

        //log accuracy for every 10 epochs
        if (!val_data.empty() && (epoch + 1) % 10 == 0) {
            float acc = get_accuracy(val_data);
            cout << "Epoch " << (epoch + 1) << " | Val Accuracy: " 
                 << (acc * 100.0f) << "%" << endl;
        }
    }
}