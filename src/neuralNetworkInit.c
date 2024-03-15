#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuralNetworkInit.h"

#include "neuralNetworkStructs.h"
#include "randomValueGenerator.h"
#include "vectorOperations.h"
#include "neuralNetworkUtility.h"


/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
void initNeuralNetwork(NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs) {

    nn->nrOfParameterNeurons = nrOfParameters;
    nn->nrOfHiddenLayers = nrOfLayers;
    nn->neuronsPerLayer = neuronsPerLayer;
    nn->nrOfOutputNeurons = nrOfOutputs;
    nn->nrOfNeurons = nrOfParameters + neuronsPerLayer*nrOfLayers + nrOfOutputs;
    nn->nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;
 
    nn->neuronActivationVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->activationParameterVector = nn->neuronActivationVector;
    nn->hiddenActivationVector = nn->neuronActivationVector + nn->nrOfParameterNeurons;
    nn->activationOutputVector = nn->neuronActivationVector + nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons;

    nn->neuronValueVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->parameterValueVector = nn->neuronValueVector;
    nn->hiddenValueVector = nn->neuronValueVector + nn->nrOfParameterNeurons;
    nn->outputValueVector = nn->neuronValueVector + nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons;

    nn->biasVector = (double *) malloc(sizeof(double) * (nn->nrOfNeurons));
    nn->weightMatrix = (double **) malloc(sizeof(double *) * (nn->nrOfNeurons-nn->nrOfOutputNeurons));

    // allocate memory for the weight matrix

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        nn->weightMatrix[i] = (double *) malloc(sizeof(double) * nrOfConnectedNeurons);
        vectorSet(nn->weightMatrix[i], 0, nrOfConnectedNeurons);
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = 0;
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronValueVector[i] = 0;
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = 0;
    }

}

/**
 * Initializes the activation functions of the neural network.
 * @param nn: the neural network to initialize the activation functions for.
 * @param activationFunction: the activation function for the hidden layers.
 * @param activationFunctionDerivative: the derivative of the activation function for the hidden layers.
 * @param lastLayerActivationFunction: the activation function for the output layer.
 * @param lastLayerActivationFunctionDerivative: the derivative of the activation function for the output layer. */
void initNeuralNetworkFunctions(NeuralNetwork * nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative, dblAdblR lastLayerActivationFunction, dblAdblR lastLayerActivationFunctionDerivative) {
    
    nn->activationFunction = activationFunction;
    nn->activationFunctionDerivative = activationFunctionDerivative;
    nn->lastLayerActivationFunction = lastLayerActivationFunction;
    nn->lastLayerActivationFunctionDerivative = lastLayerActivationFunctionDerivative;
    
}

/**
 * Initializes the weights of the neural network to a random number within "Xavier" range uniformly.
 * @param nn: the neural network to initialize the weights for.
 * @param seed: the seed for the random function */
void initWeightsXavierUniform(NeuralNetwork * nn) {

    double range;
    int nrOfConnectedNeurons;

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        range = sqrt(6.0 / (nn->nrOfParameterNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(-range, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        range = sqrt(6.0 / (nn->neuronsPerLayer + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(-range, range);
        }
    }
}

/**
*  Initializes the weights of the neural network to a random number within "Xavier" range normally.
*  @param nn: the neural network to initialize the weights for.
*  @param seed: the seed for the random function */
void initWeightsXavierNormal(NeuralNetwork * nn) {

    double range;
    int nrOfConnectedNeurons;

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        range = sqrt(2.0 / (nn->nrOfParameterNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->weightMatrix[i][j] = boxMuellerTransform(0, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        range = sqrt(2.0 / (nn->neuronsPerLayer + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->weightMatrix[i][j] = boxMuellerTransform(0, range);
        }
    }
}

/**
 * Initializes the biases of the neural network to a constant value.
 * @param nn: the neural network to initialize the biases for.
 * @param b: the constant value to initialize the biases to. */
void initBiasesConstant(NeuralNetwork * nn, double b) {
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = b;
    }
}

/**
 * Initializes the biases of the neural network to a random number within a range uniformly.
 * @param nn: the neural network to initialize the biases for.
 * @param bRange: the value range
 * @param seed: the seed for the random function */
void initBiasesRandomUniform(NeuralNetwork * nn, double minb, double maxb) {

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->biasVector[i] = randomValue(minb, maxb);
    }
}

/**
 * Initializes the weights of the neural network to a random number within a range uniformly.
 * Scaled down by a factor of 1/sqrt(nrOfParameterNeurons).
 * @param nn: the neural network to initialize the weights for.
 * @param wRange: the value range 
 * @param seed: the seed for the random function */
void initWeightsRandomUniform(NeuralNetwork * nn, double minw, double maxw) {

    for (int i = 0; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->weightMatrix[i][j] = randomValue(minw, maxw);
        }
    }

}

/**
 * Resets the neural network parameters, neurons and output.
 * @param nn: the neural network to reset. */
void resetNeuralNetwork(NeuralNetwork * nn) {

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = 0;
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronValueVector[i] = 0;
    }

}
