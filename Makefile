CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -ffast-math -fopenmp -Wall -Wextra

main: main.cpp src/math_functions.cpp src/ml_functions.cpp src/mnist.cpp src/network.cpp tests/test_basic.cpp include/math_functions.h include/ml_functions.h include/helper_functions.h include/mnist.h include/network.h tests/test_basic.h
	$(CXX) $(CXXFLAGS) main.cpp src/math_functions.cpp src/ml_functions.cpp src/mnist.cpp src/network.cpp tests/test_basic.cpp -o main

clean:
	rm -f main

.PHONY: clean

test: main
	./main

.PHONY: test
