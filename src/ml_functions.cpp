#include "../include/ml_functions.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

float relu(float x) {
    return max(0.0f, x);
}

float relu_derivative(float x) {
    return x <= 0.0f ? 0.0f : 1.0f;
}

vector<float> softmax(const vector<float>& logits) {
    if (logits.empty()) {
        throw invalid_argument("Got an empty logits array.");
    }

    float max_val = *max_element(logits.begin(), logits.end());
    vector<float> prob(logits.size());
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        prob[i] = exp(logits[i] - max_val);
        sum += prob[i];
    }

    if (sum == 0.0f) {
        throw invalid_argument("Softmax normalization failed");
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        prob[i] /= sum;
    }

    return prob;
}

float cross_entropy(const vector<float>& probs, int correct_class) {
    if (probs.empty()) {
        throw invalid_argument("Probabilities cannot be empty");
    }

    if (correct_class < 0 || correct_class >= static_cast<int>(probs.size())) {
        throw invalid_argument("Invalid class index");
    }

    float p = probs[correct_class];

    if (!isfinite(p) || p < 0.0f || p > 1.0f) {
        throw invalid_argument("Invalid probability");
    }

    constexpr float epsilon = 1e-7f;
    p = max(p, epsilon);

    return -log(p);
}
