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

    struct NeuralNetwork nn;

    nn.nrOfParameterNeurons = nrOfParameters;
    nn.nrOfHiddenLayers = nrOfLayers;
    nn.neuronsPerLayer = neuronsPerLayer;
    nn.nrOfOutputNeurons = nrOfOutputs;
    nn.nrOfWeights = nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*(nrOfLayers-1) + nrOfOutputs*neuronsPerLayer;
    nn.nrOfNeurons = nrOfParameters + neuronsPerLayer*nrOfLayers + nrOfOutputs;
    nn.nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;

    int sizeOfDouble = sizeof(double);
 
    nn.neuronVector = malloc(sizeOfDouble * (nn.nrOfNeurons));
    nn.parameterVector = nn.neuronVector;
    nn.hiddenVector = nn.neuronVector + nn.nrOfParameterNeurons;
    nn.outputVector = nn.neuronVector + nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons;
    nn.biasVector = malloc(sizeOfDouble * (nn.nrOfNeurons));
    nn.weightMatrix = malloc(sizeof(double *) * (nn.nrOfNeurons));

    int temp = nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons - nn.neuronsPerLayer;

    // allocate memory for the weight matrix

    for (int i = 0; i < temp; i++) {
        nn.weightMatrix[i] = (double *) malloc(sizeof(double) * (nn.nrOfNeurons));
    }

    for (int i = temp; i < temp + nn.neuronsPerLayer; i++) {
        nn.weightMatrix[i] = (double *) malloc(sizeof(double) * (nn.nrOfOutputNeurons));
    }
    
    
    srand(time(NULL));

    for (int i = 0; i < nn.nrOfParameterNeurons; i++) {
        nn.parameterVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfHiddenNeurons; i++) {
        nn.neuronVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfOutputNeurons; i++) {
        nn.outputVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfNeurons - nn.nrOfOutputNeurons; i++) {
        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            nn.weightMatrix[i][j] = (double) (rand() / 100000.0) + 1;
        }
    }

    for (int i = 0; i < nn.nrOfNeurons; i++) {
        nn.biasVector[i] = (double) (rand() / 100000.0) + 1;
    }

    return nn;

}

/**
 * Resets the neural network parameters, neurons and output.
 * @param nn: the neural network to reset. */
void resetNeuralNetwork(struct NeuralNetwork nn) {

    for (int i = 0; i < nn.nrOfParameterNeurons; i++) {
        nn.parameterVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfHiddenNeurons; i++) {
        nn.neuronVector[i] = 0;
    }

    for (int i = 0; i < nn.nrOfOutputNeurons; i++) {
        nn.outputVector[i] = 0;
    }
}

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(struct NeuralNetwork nn) {

    free(nn.neuronVector);

    free(nn.biasVector);

    for (int i = 0; i < nn.nrOfNeurons - nn.nrOfOutputNeurons; i++) {
        if (i > nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons - nn.nrOfOutputNeurons) {
            for (int j = 0; j < nn.nrOfOutputNeurons; j++) {
                free(nn.weightMatrix[i]);
            }
        } else {
            for (int j = 0; j < nn.neuronsPerLayer; j++) {
                free(nn.weightMatrix[i]);
            }
        }
    }
    free(nn.weightMatrix);
}