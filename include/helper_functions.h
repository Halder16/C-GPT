#pragma once

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

void print_vector(const auto& arr){
    for(auto ele: arr){
        cout<<ele<<" ";
    }
    cout<<endl;
}

void print_2d_vector(const auto& arr){
    for(auto ar:arr){
        for(auto el:ar){
            cout<<el<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}