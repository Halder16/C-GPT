#include <iostream>
#include <algorithm>

#include "./include/math_functions.h"
#include "./include/helper_functions.h"
#include "./include/mnist.h"
#include "./tests/test_basic.h"
#include "./include/network.h"

using namespace std;


void run_tests(){
    run_all_tests();
}

int main(){
    //train
    string trainImgFilename = "./archive/train-images-idx3-ubyte/train-images-idx3-ubyte";
    string trainLabelFilename = "./archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    //val
    string valImgFilename = "./archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string valLabelFilename = "./archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";


    auto train_data = get_data(trainImgFilename, trainLabelFilename);
    auto val_data = get_data(valImgFilename, valLabelFilename);

    NeuralNetwork nn;
    nn.add_layer(784, 128);
    nn.add_layer(64);
    nn.add_layer(32);
    nn.add_layer(10);

    nn.gradient_descent(train_data, 0.01f, 100, val_data);

    return 0;
}


/*
TODO: add the capability to add more layers, and for the 
neural network to handle them dynamically (buzzword alert!) 
instead of hardcoding the layer operations. 

That way, you could easily build models with 
different layers and test them
*/