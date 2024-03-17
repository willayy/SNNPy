#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuralNetworkInit.h"

#include "neuralNetworkStructs.h"
#include "randomValueGenerator.h"
#include "neuralNetworkUtility.h"

void initNeuron(Neuron * n, int nrOfConnections) {
    n->A = 0;
    n->Z = 0;
    n->conections = nrOfConnections;
    n->weights = (double *) malloc(sizeof(double) * nrOfConnections);
    n->bias = 0;
    n->activationFunctions = NULL;
    n->connectedNeurons = (Neuron **) malloc(sizeof(Neuron *) * nrOfConnections);
}

/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
void initNeuralNetwork(NeuralNetwork * nn, int nrOfInputs, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs) {

    nn->nrOfInputNeurons = nrOfInputs;
    nn->nrOfHiddenLayers = nrOfLayers;
    nn->neuronsPerLayer = neuronsPerLayer;
    nn->nrOfOutputNeurons = nrOfOutputs;
    nn->nrOfNeurons = neuronsPerLayer * nrOfLayers + nrOfOutputs + nrOfInputs;
    nn->nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;
    
    nn->inputLayerActivationFunctions = (dblAdblR *) malloc(sizeof(dblAdblR) * 2);
    nn->hiddenLayerActivationFunctions = (dblAdblR *) malloc(sizeof(dblAdblR) * 2);
    nn->outputLayerActivationFunctions = (dblAdblR *) malloc(sizeof(dblAdblR) * 2);

    nn->neurons = (Neuron **) malloc(sizeof(Neuron *) * nn->nrOfNeurons); // allocate memory for the neurons
    nn->outputLayer = nn->neurons + nn->nrOfNeurons - nn->nrOfOutputNeurons;

    // Initialize neurons
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neurons[i] = (Neuron *) malloc(sizeof(Neuron)); // allocate memory for the neurons
        int connections = numberOfConnectedNeurons(nn, i);
        initNeuron(nn->neurons[i], connections);
    }

    // Connect the neurons
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        int * connectedNeuronsIndexes = findConnectedNeuronIndexes(nn, i);
        for (int j = 0; j < nn->neurons[i]->conections; j++) {
            nn->neurons[i]->connectedNeurons[j] = nn->neurons[connectedNeuronsIndexes[j]];
        }
        free(connectedNeuronsIndexes);
    }
}

/**
 * Sets the activation function for the input layer of the neural network.
 * @param nn: the neural network to set the activation function for.
 * @param activationFunction: the activation function to set for the input layer.
 * @param activationFunctionDerivative: the derivative of the activation function to set for the input layer. */
void setInputLayerActivationFunction(NeuralNetwork *nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative) {
    nn->inputLayerActivationFunctions[0] = activationFunction;
    nn->inputLayerActivationFunctions[1] = activationFunctionDerivative;
    for (int i = 0; i < nn->nrOfInputNeurons; i++) {
        nn->neurons[i]->activationFunctions = nn->inputLayerActivationFunctions;
    }
}

/**
 * Sets the activation function for the hidden layer of the neural network.
 * @param nn: the neural network to set the activation function for.
 * @param activationFunction: the activation function to set for the hidden layer.
 * @param activationFunctionDerivative: the derivative of the activation function to set for the hidden layer. */
void setHiddenLayerActivationFunction(NeuralNetwork *nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative) {
    nn->hiddenLayerActivationFunctions[0] = activationFunction;
    nn->hiddenLayerActivationFunctions[1] = activationFunctionDerivative;
    for (int i = nn->nrOfInputNeurons; i < nn->nrOfHiddenNeurons + nn->nrOfInputNeurons; i++) {
        nn->neurons[i]->activationFunctions = nn->hiddenLayerActivationFunctions;
    }
}

/**
 * Sets the activation function for the output layer of the neural network.
 * @param nn: the neural network to set the activation function for.
 * @param activationFunction: the activation function to set for the output layer.
 * @param activationFunctionDerivative: the derivative of the activation function to set for the output layer. */
void setOutputLayerActivationFunction(NeuralNetwork *nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative) {
    nn->outputLayerActivationFunctions[0] = activationFunction;
    nn->outputLayerActivationFunctions[1] = activationFunctionDerivative;
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons; i < nn->nrOfNeurons; i++) {
        nn->neurons[i]->activationFunctions = nn->outputLayerActivationFunctions;
    }
}

/**
 * Initializes the weights of the neural network to a random number within "Xavier" range uniformly.
 * @param nn: the neural network to initialize the weights for.
 * @param seed: the seed for the random function */
void initWeightsXavierUniform(NeuralNetwork * nn) {

    double range;
    int nrOfConnectedNeurons;

    for (int i = 0; i < nn->nrOfInputNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(6.0 / (nn->nrOfInputNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = randomDouble(-range, range);
        }
    }

    for (int i = nn->nrOfInputNeurons; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(6.0 / (nn->neuronsPerLayer + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = randomDouble(-range, range);
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

    for (int i = 0; i < nn->nrOfInputNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(2.0 / (nn->nrOfInputNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = boxMuellerTransform(0, range);
        }
    }

    for (int i = nn->nrOfInputNeurons; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(2.0 / (nn->neuronsPerLayer + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = boxMuellerTransform(0, range);
        }
    }
}

/**
 * Initializes the biases of the neural network to a constant value.
 * @param nn: the neural network to initialize the biases for.
 * @param b: the constant value to initialize the biases to. */
void initBiasesConstant(NeuralNetwork * nn, double b) {
    for (int i = nn->nrOfInputNeurons; i < nn->nrOfNeurons; i++) {
        nn->neurons[i]->bias = b;
    }
}

/**
 * Initializes the biases of the neural network to a random number within a range uniformly.
 * @param nn: the neural network to initialize the biases for.
 * @param bRange: the value range
 * @param seed: the seed for the random function */
void initBiasesRandomUniform(NeuralNetwork * nn, double minb, double maxb) {

    for (int i = nn->nrOfInputNeurons; i < nn->nrOfNeurons; i++) {
        nn->neurons[i]->bias = randomDouble(minb, maxb);
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
        int nrOfConnectedNeurons = nn->neurons[i]->conections;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = randomDouble(minw, maxw);
        }
    }

}

/**
 * Resets the neural network parameters, neurons and output.
 * @param nn: the neural network to reset. */
void resetNeuralNetwork(NeuralNetwork * nn) {

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neurons[i]->A = 0;
        nn->neurons[i]->Z = 0;
    }

}
