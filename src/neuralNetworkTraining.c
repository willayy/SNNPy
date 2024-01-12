#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "sigmoid.h"
#include <stdlib.h>

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
 * Optimizes the weights and biases of the neural network by brute forcing numerical differentiation. Not very efficient.
 * @param nn The neural network.
 * @param cost The cost of the neural network on the given input and desired output.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network. */
void optimizeWeightsAndBiases(struct NeuralNetwork nn, double cost, double * input, double * desiredOutput, double lrw, double lrb, double delta) {

    double newCost = 0;
    double gradient = 0;

    // finding dCost/dw_i and dCost/db_i by brute force numerical differentiation
    for (int i = 0; i < nn.nrOfWeights; i++) {
        nn.weightMatrix[i] += delta;
        inputDataToNeuralNetwork(nn, input);
        newCost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
        nn.weightMatrix[i] -= delta;
        gradient = (double) (newCost - cost) / delta;
        nn.weightMatrix[i] -= lrw * gradient;
    }

    for (int i = 0; i < nn.nrOfLayers*nn.neuronsPerLayer; i++) {
        nn.biasVector[i] += delta;
        inputDataToNeuralNetwork(nn, input);
        newCost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
        nn.biasVector[i] -= delta;
        gradient = (double) (newCost - cost) / delta;
        nn.biasVector[i] -= lrb * gradient;
    }

}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network. */
double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb, double delta) {
    
    inputDataToNeuralNetwork(nn, input);
    
    double cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
    
    optimizeWeightsAndBiases(nn, cost, input, desiredOutput, lrw, lrb, delta);

    cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);

    return cost;
}
