#include <iostream>

#include "./include/math_functions.h"
#include "./include/helper_functions.h"
#include "./include/mnist.h"
#include "./tests/test_basic.h"

using namespace std;


void run_tests(){
    run_all_tests();
}

int main(){
    //
    // run_tests();

    string label_filename = "./archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    vector<uint8_t> label_buffer;
    auto [magic_number1, num_labels] = load_labels(label_filename, label_buffer);

    //how to check?
    if(magic_number1 == 0) cout<<"failing, noted."<<endl;
    else cout<<"loading labels worked"<<endl;

    for (size_t i = 0; i < 10; i++) {
        // Because uint8_t prints as a char, cast to int to view the numeric value
        cout << "Image #" << i << " Label: " << static_cast<int>(label_buffer[i]) << endl;
    }

    return 0;
}