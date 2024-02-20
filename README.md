# SNNPy
This is a Python module with api's to a simple C implemented configurable FFN neural network. 

The training algorithm in Beta 1.0.0 uses brute force numerical derivatives to optimize weights and biases. Release Beta 2.0.0 and onwards use backpropogation as a means to opitimize weights and biases.

This NeuralNetwork modeling software was made for learnings-sake and is not made for performance in the first hand and features no parallism or vectorization operations. Code readability also takes heavy precedence over algorithm optimization.

#### Dependencies

No external dependencies

#### Installation

- Install source code.
- (Recommended) build SNNpy-test executable to test if software works correctly in your enviorment.
- (Recommended) build SNNpy-benchmark executable to further test integrity and to benchmark software
- Build the shared libraries for the python module using CMake (recommended build tool).
- Start using python module!

#### This project was developed and tested using:

- OS: Microsoft Windows 11 version 23H2, MacOs Sonoma version 14
- Compiler: GCC 13.10.0
- Build System: CMake 3.28.1

#### Usage
