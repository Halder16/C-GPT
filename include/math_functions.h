# pragma once
#include <vector>
#include <algorithm>

using std::vector;


float vector_dot_product(const vector<float>& a, const vector<float>& b);
vector<float> vector_add(const vector<float>& a, const vector<float>& b);
vector<float> vector_subtract(const vector<float>& a, const vector<float>& b);
vector<float> vector_scale(const vector<float>& a, float f);
vector<vector<float>> to_col_matrix(const vector<float>& arr);
vector<vector<float>> mat_transpose(const vector<vector<float>>& arr);
vector<vector<float>> mat_mult(const vector<vector<float>>& a, const vector<vector<float>>& b);
vector<float> mat_vec_mult(const vector<vector<float>>& A, const vector<float>& x);
int argmax(const vector<float>& v);
