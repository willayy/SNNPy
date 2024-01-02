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
 * Backpropagates the cost through the output layer.
 * @param nn The neural network.
 * @param cost The cost of the neural network on a given input and desired output.
 * @param outputSize The size of the output array. */
void bpOutputLayer(struct NeuralNetwork nn, double * cost, int outputSize) {
    double * edges = nn.outputLayer.edges;
    double * biases = nn.outputLayer.biases;
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            edges[j] = edges[j] + cost[i];
            biases[j] = biases[j] + cost[i];
        }
        edges += nn.neuronsPerLayer;
    }
}

/**
 * Backpropagates the cost through the intermediate layers.
 * @param nn The neural network.
 * @param cost The cost of the neural network on a given input and desired output.
 * @param outputSize The size of the output array. */
void bpIntermediateLayers(struct NeuralNetwork nn, double * cost, int outputSize) {
    for (int i = 0; i < nn.nrOfLayers; i++) {
        
    }
}

/**
 * Backpropagates the cost through the neural network.
 * @param nn The neural network.
 * @param cost The cost of the neural network on a given input and desired output.
 * @param outputSize The size of the output array. */
void backPropagate(struct NeuralNetwork nn, double * cost, int outputSize) {
    bpOutputLayer(nn, cost, outputSize);
    bpIntermediateLayers(nn, cost, outputSize);
}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input of the neural network.
 * @param desiredOutput The desired output of the neural network. */
void trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput) {
    double * output = inputDataToNeuralNetwork(nn, input);
    double * cost = costFunction(output, desiredOutput, nn.nrOfOutputs);
    backPropagate(nn, cost, nn.nrOfOutputs);
}



