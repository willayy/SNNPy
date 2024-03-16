#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuralNetworkInit.h"

#include "neuralNetworkStructs.h"
#include "randomValueGenerator.h"
#include "vectorOperations.h"
#include "neuralNetworkUtility.h"

void initNeuron(Neuron * n) {
    n->A = 0;
    n->Z = 0;
    n->conections = 0;
    n->weights = NULL;
    n->bias = 0;
    n->activationFunctions = NULL;
    n->connectedNeurons = NULL;
}

void initParameter(Parameter * p) {
    p->value = 0;
    p->connectedNeurons = NULL;
}

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
    nn->nrOfNeurons = neuronsPerLayer * nrOfLayers + nrOfOutputs;
    nn->nrOfHiddenNeurons = neuronsPerLayer * nrOfLayers;
    
    nn->parameters = (Parameter **) malloc(sizeof(Parameter *) * nrOfParameters); // allocate memory for the parameters
    nn->neurons = (Neuron **) malloc(sizeof(Neuron *) * nn->nrOfNeurons); // allocate memory for the neurons

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nn->parameters[i] = (Parameter *) malloc(sizeof(Parameter)); // allocate memory for the parameters
        initParameter(nn->parameters[i]);
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neurons[i] = (Neuron *) malloc(sizeof(Neuron)); // allocate memory for the neurons
        initNeuron(nn->neurons[i]);
    }

    // Connect parameters to first layer neurons
    for (int i = 0; i < nn->nrOfParameterNeurons; i++) { 
        nn->parameters[i]->connections = neuronsPerLayer;
        nn->parameters[i]->connectedNeurons = (Neuron **) malloc(sizeof(Neuron *) * neuronsPerLayer);
        for (int j = 0; j < neuronsPerLayer; j++) {
            nn->parameters[i]->connectedNeurons[j] = nn->neurons[j];
        }
    }

    // Connect neurons to neurons
    for (int i = 0; i < nn->nrOfHiddenNeurons - nn->neuronsPerLayer; i++) {
        nn->neurons[i]->conections = neuronsPerLayer;
        nn->neurons[i]->connectedNeurons = (Neuron **) malloc(sizeof(Neuron *) * neuronsPerLayer);
        for (int j = 0; j < neuronsPerLayer; j++) {
            //TODO: fix this;
        }
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
    
    for (int i = 0; i < nn->nrOfHiddenNeurons - nn->neuronsPerLayer; i++) {
        nn->neurons[i]->activationFunctions = (dblAdblR *) malloc(sizeof(dblAdblR *) * 2);
        nn->neurons[i]->activationFunctions[0] = activationFunction;
        nn->neurons[i]->activationFunctions[1] = activationFunctionDerivative;
    }
    for (int i = nn->nrOfHiddenLayers - nn->neuronsPerLayer; i < nn->nrOfHiddenLayers; i++) {
        nn->neurons[i]->activationFunctions = (dblAdblR *) malloc(sizeof(dblAdblR *) * 2);
        nn->neurons[i]->activationFunctions[0] = lastLayerActivationFunction;
        nn->neurons[i]->activationFunctions[1] = lastLayerActivationFunctionDerivative;
    }
}

/**
 * Initializes the weights of the neural network to a random number within "Xavier" range uniformly.
 * @param nn: the neural network to initialize the weights for.
 * @param seed: the seed for the random function */
void initWeightsXavierUniform(NeuralNetwork * nn) {

    double range;
    int nrOfConnectedNeurons;

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(6.0 / (nn->nrOfParameterNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = randomDouble(-range, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
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

    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {
        nrOfConnectedNeurons = nn->neurons[i]->conections;
        range = sqrt(2.0 / (nn->nrOfParameterNeurons + nrOfConnectedNeurons));
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            nn->neurons[i]->weights[j] = boxMuellerTransform(0, range);
        }
    }

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
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
    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons; i++) {
        nn->neurons[i]->bias = b;
    }
}

/**
 * Initializes the biases of the neural network to a random number within a range uniformly.
 * @param nn: the neural network to initialize the biases for.
 * @param bRange: the value range
 * @param seed: the seed for the random function */
void initBiasesRandomUniform(NeuralNetwork * nn, double minb, double maxb) {

    for (int i = nn->nrOfParameterNeurons; i < nn->nrOfNeurons; i++) {
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
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neurons[i] = 0;
    }

}
