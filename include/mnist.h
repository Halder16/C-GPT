#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <array>

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::cerr;
using std::vector;
using std::array;


struct Train_data {
    vector<float> image;
    int label;
};


uint32_t swap_endian(uint32_t val);
array<uint32_t,4> load_images(const string fileName, vector<uint8_t>& pixel_buffer);
array<uint32_t,2> load_labels(const string fileName, vector<uint8_t>& label_buffer);
vector<float> normalize_image_pixels(const vector<uint8_t>& pixel_buffer);
void print_ascii(const vector<uint8_t>& pixel_buffer, const uint32_t num_images, 
    const uint32_t rows, const uint32_t cols);
vector<Train_data> get_data(const string imgFilename, const string labelFilename);