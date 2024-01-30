#include "neuralNetworkStructs.h"
#include "nodeUtility.h"
#include <stdlib.h>

/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron) {

    if (neuron < nn.nrOfParameterNeurons) {
        return nn.neuronsPerLayer;
    }
    else if (neuron < nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons - nn.neuronsPerLayer) {
        return nn.neuronsPerLayer;
    }
    else if (neuron < nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons) {
        return nn.nrOfOutputNeurons;
    }
}

/**
 * Returns pointer to a neuron in the neural network, cant be a output neuron.
 * @param nn The neural network.
 * @param node The neuron. */
double * findNeuron(struct NeuralNetwork nn, int neuron) {
    return & nn.neuronVector[neuron];
}

/**
 * returns a vector of the nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
double * findConnectedNeurons(struct NeuralNetwork nn, int neuron) {

    if (neuron < nn.nrOfParameterNeurons) {
        return nn.neuronVector + nn.nrOfParameterNeurons;
    }
    else if (neuron < nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons - nn.neuronsPerLayer) {
        return nn.hiddenVector + (1 + neuron / nn.neuronsPerLayer) * nn.neuronsPerLayer;
    }
    else if (neuron < nn.nrOfParameterNeurons + nn.nrOfHiddenNeurons) {
        return nn.outputVector;
    }
}

/**
 * returns a vector of the weights connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected weights of (node 0 is the top parameter node).
 * @return A vector of the connected weights. */
double * findConnectedWeights(struct NeuralNetwork nn, int neuron) {
    return nn.weightMatrix[neuron];
}

/** 
 * Returns a copy vector of the activation values of the neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getActivationValues(struct NeuralNetwork nn) {
    double * activationValues = malloc(sizeof(double) * nn.nrOfNeurons);
    for (int i = 0; i < nn.nrOfNeurons; i++) {
        activationValues[i] = nn.neuronVector[i];
    }
    return activationValues;
}