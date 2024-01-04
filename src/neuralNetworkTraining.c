#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include <stdlib.h>

/**
 * Calculates the cost of the neural network on a given input and desired output. 
 * Calculating how good bad the current biases and edges are for a certain input.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double * costFunction(double * output, double * desiredOutput, int outputSize) {

    double cost[outputSize];

    for (int i = 0; i < outputSize; i++) {
        cost[i] = (output[i] - desiredOutput[i]);
    }

    return cost;
}

/**
 * Backpropagates the cost through the neural network.
 * @param nn The neural network.
 * @param cost The cost of the neural network on a given input and desired output.
 * @param outputSize The size of the output array. */
void backPropagate(struct NeuralNetwork nn, double * cost, int outputSize) {
    
}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input of the neural network.
 * @param desiredOutput The desired output of the neural network. */
void trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput) {
    
}



