# SNNPy
This is a Python module with api's to a simple C implemented configurable FFN neural network. 

The training algorithm in Beta 1.0.0 uses brute force numerical derivatives to optimize weights and biases. Release Beta 2.0.0 and onwards use backpropogation as a means to opitimize weights and biases.

This NeuralNetwork model is not made for performance in the first hand and features no parallism or vectorization operations. Code readability also takes heavy precedence over algorithm optimization.

#### Dependencies

No external dependencies

#### Installation

- Its recommended to build the shared libraries for the python module from scrath using the CMake MakeFile

#### This project was developed and tested using:

- OS: Microsoft Windows 11 version 10.0.22621 and later
- Compiler: GCC 13.10.0
- Build System: CMake 3.28.1

#### Usage
