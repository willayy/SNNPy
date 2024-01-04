#include <stdlib.h>
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
    neuralNetwork.nrOfLayers = nrOfLayers;
    neuralNetwork.neuronsPerLayer = neuronsPerLayer;
    neuralNetwork.nrOfOutputs = nrOfOutputs;

    int sizeOfDouble = sizeof(double);
    neuralNetwork.parameterVector = (double*) malloc(sizeOfDouble * nrOfParameters);
    neuralNetwork.neuronVector = (double*) malloc(sizeOfDouble * (nrOfLayers*neuronsPerLayer));
    neuralNetwork.outputVector = (double*) malloc(sizeOfDouble * nrOfOutputs);
    neuralNetwork.biasVector = (double*) malloc(sizeOfDouble * (nrOfLayers*neuronsPerLayer));
    neuralNetwork.weightMatrix = (double*) malloc(sizeOfDouble * (nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*nrOfLayers + nrOfOutputs*neuronsPerLayer));

    int totalNrOfWeights = nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*nrOfLayers + nrOfOutputs*neuronsPerLayer;
    
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

    for (int i = 0; i < totalNrOfWeights; i++) {
        neuralNetwork.weightMatrix[i] = (double) rand() / rand();
    }

    for (int i = 0; i < nrOfLayers*neuronsPerLayer; i++) {
        neuralNetwork.biasVector[i] = (double) rand() / rand();
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

    for (int i = 0; i < nn.nrOfLayers*nn.neuronsPerLayer; i++) {
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