#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuralNetworkInit.h"

#include "neuralNetworkStructs.h"
#include "randomValueGenerator.h"


/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
void initNeuralNetwork(struct NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs, dblA activationFunction, dblA activationFunctionDerivative, dblA lastLayerActivationFunction, dblA lastLayerActivationFunctionDerivative) {

    nn->activationFunction = activationFunction;
    nn->activationFunctionDerivative = activationFunctionDerivative;
    nn->lastLayerActivationFunction = lastLayerActivationFunction;
    nn->lastLayerActivationFunctionDerivative = lastLayerActivationFunction;

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

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nn->biasVector[i] = 0;
    }

}
/**
 * Initializes the weights of the neural network to a random number within "Xavier" range uniformly.
 * @param nn: the neural network to initialize the weights for.
 * @param seed: the seed for the random function */
void initWeightsXavierUniform(struct NeuralNetwork * nn) {

    double range = sqrt(6.0 / (nn->nrOfParameterNeurons + nn->neuronsPerLayer));

    for (int i = 0; i < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i++) {
        for (int j = 0; j < nn->neuronsPerLayer; j++) {
            nn->weightMatrix[i][j] = randomValue(-range, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i < nn->nrOfNeurons; i++) {
        for (int j = 0; j < nn->nrOfOutputNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(-range, range);
        }
    }
}

/**
*  Initializes the weights of the neural network to a random number within "Xavier" range normally.
*  @param nn: the neural network to initialize the weights for.
*  @param seed: the seed for the random function */
void initWeightsXavierNormal(struct NeuralNetwork * nn) {

    double range = sqrt(2.0 / (nn->nrOfParameterNeurons + nn->neuronsPerLayer));

    for (int i = 0; i < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i++) {
        for (int j = 0; j < nn->neuronsPerLayer; j++) {
            nn->weightMatrix[i][j] = boxMuellerTransform(0, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i < nn->nrOfNeurons; i++) {
        for (int j = 0; j < nn->nrOfOutputNeurons; j++) {
            nn->weightMatrix[i][j] = boxMuellerTransform(0, range);
        }
    }
}

/**
 * Initializes the biases of the neural network to a constant value.
 * @param nn: the neural network to initialize the biases for.
 * @param b: the constant value to initialize the biases to. */
void initBiasesConstant(struct NeuralNetwork * nn, double b) {
    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = b;
    }
}

/**
 * Initializes the biases of the neural network to a random number within a range uniformly.
 * @param nn: the neural network to initialize the biases for.
 * @param bRange: the value range
 * @param seed: the seed for the random function */
void initBiasesRandomUniform(struct NeuralNetwork * nn, double * bRange) {

    double minb = bRange[0];
    double maxb = bRange[1];

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = randomValue(minb, maxb);
    }
}

/**
 * Initializes the weights of the neural network to a random number within a range uniformly.
 * Scaled down by a factor of 1/sqrt(nrOfParameterNeurons).
 * @param nn: the neural network to initialize the weights for.
 * @param wRange: the value range 
 * @param seed: the seed for the random function */
void initWeightsRandomUniform(struct NeuralNetwork * nn, double * wRange) {

    double minw = wRange[0];
    double maxw = wRange[1];

    for (int i = 0; i < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i++) {
        for (int j = 0; j < nn->neuronsPerLayer; j++) {
            nn->weightMatrix[i][j] = randomValue(minw, maxw) / (1/sqrt(nn->nrOfParameterNeurons));
        }
    }

    for (int i = nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons; i < nn->nrOfNeurons; i++) {
        for (int j = 0; j < nn->nrOfOutputNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(minw, maxw) / (1/sqrt(nn->nrOfParameterNeurons));
        }
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