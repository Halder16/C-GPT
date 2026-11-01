#include "../include/mnist.h"


/*
    train-images-idx3-ubyte : training images
    train-labels-idx3-ubyte : training labels

    t10k-images-idx3-ubyte : test images
    t10k-labels-idx1-ubyte : test labels
*/


// helper
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
           ((val >> 8)  & 0x0000FF00) |
           ((val << 8)  & 0x00FF0000) |
           ((val << 24) & 0xFF000000);
}


array<uint32_t,4> load_images(const string fileName, vector<uint8_t>& pixel_buffer){
    //this works, i think

    ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << fileName << std::endl;
        return {};
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    // Read 16-byte header fields
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic_number = swap_endian(magic_number);

    if(magic_number != 2051) {
        cerr<<"No images in here."<<endl;
        return {};
    }

    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    // Calculate total size of the raw pixel data buffer
    size_t total_pixels = static_cast<size_t>(num_images) * rows * cols;
    pixel_buffer.resize(total_pixels);

    // Read all pixel data directly into the vector memory
    file.read(reinterpret_cast<char*>(pixel_buffer.data()), total_pixels);

    if (file) {
        cout << "Successfully loaded " << total_pixels << " pixels." << std::endl;

    } else {
        cerr << "Error: Only read " << file.gcount() << " bytes." << std::endl;
    }

    return {magic_number, num_images, rows, cols};
}

array<uint32_t,2> load_labels(const string fileName, vector<uint8_t>& label_buffer){
    //

    ifstream file(fileName, std::ios::binary);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << fileName << std::endl;
        return {};
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic_number = swap_endian(magic_number);

    if(magic_number != 2049) {
        cerr<<"No images in here."<<endl;
        return {};
    }

    num_labels = swap_endian(num_labels);

    cout << "Label Magic Number: " << magic_number << endl;
    cout << "Total Labels: " << num_labels << endl;

    label_buffer.resize(num_labels);

    // Read the sequential label bytes directly into memory
    file.read(reinterpret_cast<char*>(label_buffer.data()), num_labels);

    if (file) {
        cout << "Successfully loaded labels." << endl;
    }   
    
    return {magic_number, num_labels};
}






// int main() {
//     string image_filename = "../archive/train-images-idx3-ubyte/train-images-idx3-ubyte";
    
//     vector<uint8_t> pixel_buffer;

//     auto [magic_number, num_images, rows, cols] = load_images(image_filename, pixel_buffer);

//     //how to check?
//     if(magic_number == 0){
//         //reading failed, so
//         cout<<"failing noted"<<endl;
//     }else{
//         cout<<"loading images worked!"<<endl;
//     }

//     // for(int i=0;i<2;i++){
//     //     cout<<"image "<<i<<":"<<endl;
//     //     for(int r=0;r<rows;r++){
//     //         for(int c=0;c<cols;c++){
//     //             int idx = (i * rows * cols) + (r * cols) + c;
//     //             cout<<(int)pixel_buffer[idx]<<" ";
//     //         }
//     //         cout<<endl; //one image ends?
//     //     }
//     //     cout<<endl<<endl;
//     // }    


//     string label_filename = "../archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

//     vector<uint8_t> label_buffer;
//     auto [magic_number1, num_labels] = load_labels(label_filename, label_buffer);

//     //how to check?
//     if(magic_number1 == 0) cout<<"failing, noted."<<endl;
//     else cout<<"loading labels worked"<<endl;

//     // for (size_t i = 0; i < 10; i++) {
//     //     // Because uint8_t prints as a char, cast to int to view the numeric value
//     //     cout << "Image #" << i << " Label: " << static_cast<int>(label_buffer[i]) << endl;
//     // }

//     cout<<"images: "<<num_images<<", labels: "<<num_labels<<endl;

//     return 0;
// }
