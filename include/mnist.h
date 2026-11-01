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


uint32_t swap_endian(uint32_t val);
array<uint32_t,4> load_images(const string fileName, vector<uint8_t>& pixel_buffer);
array<uint32_t,2> load_labels(const string fileName, vector<uint8_t>& label_buffer);

