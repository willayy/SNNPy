# SNNPy
This is a Python module with api's to a simple C implemented configurable layer FFN neural network. This repo contains the C source code in SNNpy/src and the Python module source code in SNNPy/snnpy. For this neural network i used the sigmoid function as the activation function. The training algorithm in Beta 1.0.0 uses brute force numerical derivatives to optimize weights and biases. Release Beta 2.0.0 will use backpropogation as a means to opitimize weights and biases.

This NeuralNetwork model is not made for performance in the first hand and features no parallism or vectorization operations. Code readability also takes heavy precedence over algorithm optimization.

#### Dependencies

- exceptions4c docs and more at https://github.com/guillermocalvo/exceptions4c (this project contains a stripped down verison)

#### This project was developed and tested using:

- OS: Microsoft Windows 11
- OS Version: 10.0.22621
- Compiler: GCC 
- Compiler Version: 13.10.0
- Build System: CMake 0.0.17 (VSCode extension)

#### Features
