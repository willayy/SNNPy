#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunction.h"
#include "sigmoid.h"
#include <stdlib.h>

/**
 * Backpropogates the error through the neural network and optimizes the weights and biases.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void backPropogate(struct NeuralNetwork nn, double * desiredOutput, double lrw, double lrb) {

    double * derivatives = malloc(sizeof(double)*nn.neuronsPerLayer);
    
    
}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network. */
double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb, double delta) {
    
    inputDataToNeuralNetwork(nn, input);
    
    double cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
    
    backPropogate(nn, desiredOutput, lrw, lrb);

    return cost;
}
