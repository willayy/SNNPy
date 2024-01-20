#include "costFunction.h"

/**
 * Calculates the cost of the neural network on a given input and desired output. 
 * Calculating how good bad the current biases and edges are for a certain input.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double costFunction(double * output, double * desiredOutput, int outputSize) {

    double cost = 0;

    for (int i = 0; i < outputSize; i++) {
        cost += (output[i]-desiredOutput[i])*(output[i]-desiredOutput[i]);
    }

    return cost;
}

/**
 * Calculates the derivative of the cost function for a single output.
 * @param output The output of the neural network neuron.
 * @param desiredOutput The desired output of the neural network.
 * @return The derivative of the cost function. */
double costFunctionDerivative(double output, double desiredOutput) {
    return 2 * (output - desiredOutput);
}