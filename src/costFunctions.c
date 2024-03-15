#include "costFunctions.h"
#include <math.h>
#include <stdlib.h>

/**
 * Calculates the cost of the neural network on a given input and desired output. 
 * Calculating how good bad the current biases and edges are for a certain input.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double sqrCostFunction(double * output, double * desiredOutput, int outputSize) {

    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    double cost = 0;
    for (int i = 0; i < outputSize; i++) {
        double error = output[i] - desiredOutput[i];
        cost += error * error;
    }
    return 0.5 * cost;
}

/**
 * Calculates the derivative of the cost function for a single output.
 * @param output The output of the neural network neuron.
 * @param desiredOutput The desired output of the neural network.
 * @return The derivative of the cost function. */
double sqrCostFunctionDerivative(double output, double desiredOutput) {
    return (output-desiredOutput);
}

/**
 * Calculates the cross entropy cost of the neural network on a given input and desired output.
 * Use this loss function with softmax activation.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize) {

    // Note: expected outputs are expected to all be either 0 or 1
    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    double cost = 0;
    for (int i = 0; i < outputSize; i++) {
        double x = output[i];
        double y = desiredOutput[i];
        double v = (y == 1) ? -log(x) : -log(1 - x);
        cost += isnan(v) ? 0 : v;
    }
    return cost;
}

/**
 * Calculates the derivative of the cross entropy cost function for a single output.
 * Use this loss function with softmax activation.
 * @param output The output of the neural network neuron.
 * @param desiredOutput The desired output of the neural network.
 * @return The derivative of the cost function. */
double crossEntropyCostFunctionDerivative(double output, double desiredOutput ) {
    double x = output;
    double y = desiredOutput;
    if (x == 0 || x == 1) { return 0; }
    return (-x + y) / (x * (x - 1));
}