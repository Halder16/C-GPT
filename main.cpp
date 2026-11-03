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
    //
    NeuralNetwork nn;
    vector<float> image(784, 0.5f); // Example input
    vector<float> predictions = nn.forward_prop(image);

    int digit = max_element(
        predictions.begin(),
        predictions.end()
    ) - predictions.begin();

    cout << "Predicted digit: " << digit << endl;
    cout<<predictions.size()<<endl;
    cout<<accumulate(predictions.begin(), predictions.end(),0.0f)<<endl;

    return 0;
}