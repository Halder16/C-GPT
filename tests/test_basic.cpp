#include "./test_basic.h"

using namespace std;

bool approx_equal(float a, float b, float tolerance = 1e-5f) {
    return fabs(a - b) < tolerance;
}


// ==========================================
// TEST 1: ReLU
// ==========================================

void test_relu() {
    assert(relu(5.0f) == 5.0f);
    assert(relu(-5.0f) == 0.0f);
    assert(relu(0.0f) == 0.0f);
    assert(relu(-0.001f) == 0.0f);

    cout << "[PASS] ReLU\n";
}


// ==========================================
// TEST 2: ReLU Derivative
// ==========================================

void test_relu_derivative() {
    assert(relu_derivative(5.0f) == 1.0f);
    assert(relu_derivative(-5.0f) == 0.0f);
    assert(relu_derivative(0.0f) == 0.0f);

    cout << "[PASS] ReLU derivative\n";
}


// ==========================================
// TEST 3: Softmax
// ==========================================

void test_softmax() {

    // Ordinary inputs
    vector<float> x1 = {2.0f, 1.0f, 0.0f};
    vector<float> p1 = softmax(x1);

    assert(approx_equal(p1[0], 0.665241f));
    assert(approx_equal(p1[1], 0.244728f));
    assert(approx_equal(p1[2], 0.090031f));

    // All zeros: probabilities should be equal
    vector<float> x2 = {0.0f, 0.0f, 0.0f};
    vector<float> p2 = softmax(x2);

    for (float p : p2) {
        assert(approx_equal(p, 1.0f / 3.0f));
    }

    // Negative inputs
    vector<float> x3 = {-2.0f, -1.0f, -3.0f};
    vector<float> p3 = softmax(x3);

    float sum = 0.0f;

    for (float p : p3) {
        assert(isfinite(p));
        assert(p >= 0.0f && p <= 1.0f);
        sum += p;
    }

    assert(approx_equal(sum, 1.0f));

    // Large inputs: test numerical stability
    vector<float> x4 = {1000.0f, 1001.0f, 1002.0f};
    vector<float> p4 = softmax(x4);

    sum = 0.0f;

    for (float p : p4) {
        assert(isfinite(p));
        sum += p;
    }

    assert(approx_equal(sum, 1.0f));

    // Empty input should throw
    bool exception_thrown = false;

    try {
        vector<float> empty;
        softmax(empty);
    }
    catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    cout << "[PASS] Softmax\n";
}


// ==========================================
// TEST 4: Cross Entropy
// ==========================================

void test_cross_entropy() {

    vector<float> probs = {0.1f, 0.7f, 0.2f};

    // Correct class = 1, probability = 0.7
    float loss = cross_entropy(probs, 1);

    assert(approx_equal(loss, -log(0.7f)));

    // Perfect prediction should have approximately zero loss
    vector<float> perfect = {0.0f, 1.0f, 0.0f};

    loss = cross_entropy(perfect, 1);

    assert(approx_equal(loss, 0.0f));

    // Invalid class index
    bool exception_thrown = false;

    try {
        cross_entropy(probs, 5);
    }
    catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    // Negative class index
    exception_thrown = false;

    try {
        cross_entropy(probs, -1);
    }
    catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    // Empty probability vector
    exception_thrown = false;

    try {
        vector<float> empty;
        cross_entropy(empty, 0);
    }
    catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    cout << "[PASS] Cross entropy\n";
}


// ==========================================
// TEST 5: Matrix Multiplication
// ==========================================

void test_mat_mult() {

    // 2x3 multiplied by 3x2
    vector<vector<float>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };

    vector<vector<float>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    vector<vector<float>> C = mat_mult(A, B);

    // Expected result:
    // 58   64
    // 139  154

    assert(C.size() == 2);
    assert(C[0].size() == 2);

    assert(approx_equal(C[0][0], 58.0f));
    assert(approx_equal(C[0][1], 64.0f));
    assert(approx_equal(C[1][0], 139.0f));
    assert(approx_equal(C[1][1], 154.0f));

    // Square matrix multiplication
    vector<vector<float>> X = {
        {1, 2},
        {3, 4}
    };

    vector<vector<float>> Y = {
        {5, 6},
        {7, 8}
    };

    vector<vector<float>> Z = mat_mult(X, Y);

    assert(approx_equal(Z[0][0], 19.0f));
    assert(approx_equal(Z[0][1], 22.0f));
    assert(approx_equal(Z[1][0], 43.0f));
    assert(approx_equal(Z[1][1], 50.0f));

    // Incompatible dimensions
    bool exception_thrown = false;

    try {
        vector<vector<float>> P = {
            {1, 2, 3},
            {4, 5, 6}
        };

        vector<vector<float>> Q = {
            {1, 2},
            {3, 4}
        };

        mat_mult(P, Q);
    }
    catch (const invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    cout << "[PASS] Matrix multiplication\n";
}


// ==========================================
// RUN ALL TESTS
// ==========================================

void run_all_tests() {

    cout << "\nRunning tests...\n\n";

    test_relu();
    test_relu_derivative();
    test_softmax();
    test_cross_entropy();
    test_mat_mult();

    cout << "\nAll tests passed!\n";
}
