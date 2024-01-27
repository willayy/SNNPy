#include <stdlib.h>
#include <stdio.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include <time.h>

/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs) {

    struct NeuralNetwork neuralNetwork;

    neuralNetwork.nrOfParameters = nrOfParameters;
    neuralNetwork.nrOfHiddenLayers = nrOfLayers;
    neuralNetwork.neuronsPerLayer = neuronsPerLayer;
    neuralNetwork.nrOfOutputs = nrOfOutputs;
    neuralNetwork.nrOfWeights = nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*nrOfLayers + nrOfOutputs*neuronsPerLayer;
    neuralNetwork.weightsPerLayer = neuronsPerLayer * neuronsPerLayer;
    neuralNetwork.nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;

    int sizeOfDouble = sizeof(double);

    neuralNetwork.parameterVector = malloc(sizeOfDouble * nrOfParameters);
    neuralNetwork.neuronVector = malloc(sizeOfDouble * (nrOfLayers*neuronsPerLayer));
    neuralNetwork.outputVector = malloc(sizeOfDouble * nrOfOutputs);
    neuralNetwork.biasVector = malloc(sizeOfDouble * (nrOfLayers*neuronsPerLayer));
    neuralNetwork.weightMatrix = malloc(sizeOfDouble * (nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*nrOfLayers + nrOfOutputs*neuronsPerLayer));
    
    srand(time(NULL));

    for (int i = 0; i < nrOfParameters; i++) {
        neuralNetwork.parameterVector[i] = 0;
    }

    for (int i = 0; i < nrOfLayers*neuronsPerLayer; i++) {
        neuralNetwork.neuronVector[i] = 0;
    }

    for (int i = 0; i < nrOfOutputs; i++) {
        neuralNetwork.outputVector[i] = 0;
    }

    for (int i = 0; i < neuralNetwork.nrOfWeights; i++) {
        // neuralNetwork.weightMatrix[i] = (double) rand() / rand();
        neuralNetwork.weightMatrix[i] = (double) (rand() / 10000.0 + 1.0);
    }

    for (int i = 0; i < nrOfLayers*neuronsPerLayer; i++) {
        // neuralNetwork.biasVector[i] = (double) rand() / rand();
        neuralNetwork.biasVector[i] = (double) (rand() / 10000.0 + 1.0);
    }

    return neuralNetwork;

}

/**
 * Resets the neural network parameters, neurons and output.
 * @param nn: the neural network to reset. */
void resetNeuralNetwork(struct NeuralNetwork nn) {
    for (int i = 0; i < nn.nrOfParameters; i++) {
        nn.parameterVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfHiddenLayers*nn.neuronsPerLayer; i++) {
        nn.neuronVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfOutputs; i++) {
        nn.outputVector[i] = 0;
    }
}

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(struct NeuralNetwork nn) {
    free(nn.parameterVector);
    free(nn.neuronVector);
    free(nn.outputVector);
    free(nn.biasVector);
    free(nn.weightMatrix);
}