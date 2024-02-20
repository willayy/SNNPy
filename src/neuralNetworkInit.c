#include <stdlib.h>
#include <time.h>

#include "neuralNetworkInit.h"

#include "neuralNetworkStructs.h"
#include "randomValueGenerator.h"


/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
void initNeuralNetwork(struct NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs, double * wRange, double * bRange, unsigned int seed) {

    nn->nrOfParameterNeurons = nrOfParameters;
    nn->nrOfHiddenLayers = nrOfLayers;
    nn->neuronsPerLayer = neuronsPerLayer;
    nn->nrOfOutputNeurons = nrOfOutputs;
    nn->nrOfWeights = nrOfParameters*neuronsPerLayer + neuronsPerLayer*neuronsPerLayer*(nrOfLayers-1) + nrOfOutputs*neuronsPerLayer;
    nn->nrOfNeurons = nrOfParameters + neuronsPerLayer*nrOfLayers + nrOfOutputs;
    nn->nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;
 
    nn->neuronActivationVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->neuronValueVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->parameterVector = nn->neuronActivationVector;
    nn->hiddenVector = nn->neuronActivationVector + nn->nrOfParameterNeurons;
    nn->outputVector = nn->neuronActivationVector + nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons;
    nn->biasVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->weightMatrix = (double **) malloc(sizeof(double *) * (nn->nrOfNeurons));

    // allocate memory for the weight matrix

    for (int i = 0; i < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i++) {
        nn->weightMatrix[i] = (double *) malloc(sizeof(double) * (nn->neuronsPerLayer));
    }

    for (int i = nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i < nn->nrOfNeurons; i++) {
        nn->weightMatrix[i] = (double *) malloc(sizeof(double) * (nn->nrOfOutputNeurons));
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = 0;
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronValueVector[i] = 0;
    }

    // Set the seed for the random number generator
    setSeed(seed);
    double minw = wRange[0];
    double maxw = wRange[1];
    double minb = bRange[0];
    double maxb = bRange[1];

    // Randomize the weights and biases

    for (int i = 0; i < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i++) {
        for (int j = 0; j < nn->neuronsPerLayer; j++) {
            nn->weightMatrix[i][j] = randomValue(minw, maxw);
        }
    }

    for (int i = nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i < nn->nrOfNeurons; i++) {
        for (int j = 0; j < nn->nrOfOutputNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(minw, maxw);
        }
    }

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nn->biasVector[i] = 0;
    }

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = randomValue(minb, maxb);
    }

}

/**
 * Resets the neural network parameters, neurons and output.
 * @param nn: the neural network to reset. */
void resetNeuralNetwork(struct NeuralNetwork * nn) {

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = 0;
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronValueVector[i] = 0;
    }

}

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(struct NeuralNetwork * nn) {

    free(nn->neuronActivationVector);

    free(nn->biasVector);

    free(nn->neuronValueVector);

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        free(nn->weightMatrix[i]);
    }
    
    free(nn->weightMatrix);

    free(nn);
}