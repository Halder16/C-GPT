#pragma once

#include <vector>

using std::vector;

float relu(float x);
float relu_derivative(float x);
vector<float> softmax(const vector<float>& logits);
float cross_entropy(const vector<float>& probs, int correct_class);
