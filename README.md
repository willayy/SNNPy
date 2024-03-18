# SNNPy
This is a Python module with api's to a C implemented configurable FFN neural network. 

This NeuralNetwork modeling software was made for learnings-sake and is not made for performance in the first hand. It does not feature parallelism, hardware accelaration or any usage of optimized BLAS algorithms. 
And such it is a pretty naive implementation of a neural network.

The neural network has the following features
- Xavier initiation of weights using box mueller transform to generate normally distributed random variables
- Uniform random initialization of weights
- Uniform random initialization of weights
- Simple forward propogation algorithm
- Simple back propogation algorithm
- Batch training
- Batch randomzitaion (ensuring that the training examples arent always in the same order)
- 4 activation functions (linear, sigmoid, ReLU, TanH)
- 2 cost functions (cross entropy cost, and mean square cost)
- Configurable learning rates
- Fully configurable feed forward network structures (only requirement being there should be at least 1 input, 1 hidden layer neuron and 1 output neuron)
- Gradient batch averaging

#### Dependencies

CMake is heavily recommended as the build tool for this source code.
No other external dependencies.

#### Installation/Building

- Download source code.
- (Recommended) build SNNpy-test executable to test if software works correctly in your enviorment.
- (Recommended) build SNNpy-benchmark executable to further test integrity and to benchmark software
- Build the shared libraries for the python module using CMake (recommended build tool).
- Start using python module!

#### This project was developed and tested using:

- Tested OS: Microsoft Windows 11 version 23H2, MacOs Sonoma version 14
- Tested compilers: GCC 13.10.0, Clang 15.0.0
- Build System: CMake 3.28.1

#### Usage
(PLACEHOLDER COMING SOON)
