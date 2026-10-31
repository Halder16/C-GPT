CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra

main: main.cpp src/math_functions.cpp src/ml_functions.cpp tests/test_basic.cpp include/math_functions.h include/ml_functions.h include/helper_functions.h tests/test_basic.h
	$(CXX) $(CXXFLAGS) main.cpp src/math_functions.cpp src/ml_functions.cpp tests/test_basic.cpp -o main

clean:
	rm -f main

.PHONY: clean

test: main
	./main

.PHONY: test
