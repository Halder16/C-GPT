using namespace std;

#include "../include/math_functions.h"
#include <stdexcept>
#include <cmath>


float vector_dot_product(const vector<float>& a, const vector<float>& b){
    // can i assume they will be of same lenght? - no, right?
    if(a.size()!=b.size()){
        throw invalid_argument("input argument vector sizes don't match!");
    }
    int n = a.size();
    float res = 0.0f;

    for(int i=0;i<n;i++){
        res += a[i]*b[i];
    }

    return res;
} 

vector<float> vector_add(const vector<float>& a, const vector<float>& b){
    if(a.size() != b.size()) throw invalid_argument("input argument vector sizes don't match!");

    int n = a.size();
    vector<float> res(n);
    for(int i=0;i<n;i++){
        res[i] = a[i] + b[i];
    }

    return res;
}

vector<float> vector_subtract(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Vector sizes must match");
    }

    vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

vector<float> vector_scale(const vector<float>& a, float f){
    vector<float> res;
    for(auto e:a){
        res.push_back(f*e);
    }

    return res;
}

vector<vector<float>> to_col_matrix(const vector<float>& arr){
    int n = arr.size();
    vector<vector<float>> res(n, vector<float> (1));
    for(int i=0;i<n;i++){
        res[i][0] = arr[i];
    }

    return res;
}

vector<vector<float>> mat_transpose(const vector<vector<float>>& arr){
    if(arr.empty() || arr[0].empty()) throw invalid_argument("Matrix cannot be empty.");

    int n = arr.size();
    int m = arr[0].size();

    for (const auto& row : arr) {
        if ((int)row.size() != m) {
            throw invalid_argument("Matrix rows must have equal lengths");
        }
    }

    vector<vector<float>> res(m, vector<float> (n));

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            res[j][i] = arr[i][j];
        }
    }

    return res;
}

vector<vector<float>> mat_mult(const vector<vector<float>>& a, const vector<vector<float>>& b){
    if(a.empty() || b.empty() || a[0].empty() || b[0].empty()) 
        throw invalid_argument("Matrices to be multiplied can't be empty");

    if(a[0].size() != b.size()) throw invalid_argument("Invalid input sizes. Matrix mult not possible.");

    int n = a.size();
    int l = a[0].size();

    int m = b[0].size();
    vector<vector<float>> res(n, vector<float> (m, 0.0f));

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            for(int k=0;k<l;k++){
                res[i][j] += a[i][k]*b[k][j];
            }
        }
    }

    return res;
}

vector<float> mat_vec_mult(
    const vector<vector<float>>& A,
    const vector<float>& x) {
    if (A.empty() || A[0].size() != x.size()) {
        throw invalid_argument("Incompatible dimensions");
    }

    vector<float> result(A.size(), 0.0f);

    for (int i = 0; i < (int)A.size(); i++) {
        for (int j = 0; j < (int)x.size(); j++) {
            result[i] += A[i][j] * x[j];
        }
    }

    return result;
}


int argmax(const vector<float>& v){
    return max_element(v.begin(), v.end()) - v.begin();
}