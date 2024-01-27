#include "neuralNetworkStructs.h"
#include "nodeUtility.h"
#include <stdlib.h>

int getNeuronLayer(struct NeuralNetwork nn, int node) {
    int isParamNode = node <= nn.nrOfParameters;
    int isHiddenNode = node > nn.nrOfParameters && node <= (nn.nrOfHiddenNeurons - nn.neuronsPerLayer);
    int isHiddenToOutputNode = node > nn.nrOfHiddenNeurons - nn.neuronsPerLayer && node <= nn.nrOfHiddenNeurons;

    if (isParamNode) {
        return 1;
    }

    else if (isHiddenNode) {
        return 2;
    }

    else if (isHiddenToOutputNode) {
        return 3;
    }
}

/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron) {
    int nodeLayer = getNeuronLayer(nn, neuron);
    if (nodeLayer == 1) {
        return nn.neuronsPerLayer;
    }
    else if (nodeLayer == 2) {
        return nn.neuronsPerLayer;
    }
    else if (nodeLayer == 3) {
        return nn.nrOfOutputs;
    }
}

/**
 * Returns a vector of the activation values of the neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getNeuronActivationValues(struct NeuralNetwork nn) {

    int nrOfNeurons = nn.nrOfParameters + nn.nrOfHiddenNeurons + nn.nrOfOutputs;

    double * neuronActivationValues = malloc(sizeof(double)*nrOfNeurons);

    for (int i = 0; i < nrOfNeurons; i++) {

        int nodeLayer = getNeuronLayer(nn, i);

        if (nodeLayer == 1) {
            neuronActivationValues[i] = nn.parameterVector[i];
        }
        else if (nodeLayer == 2) {
            neuronActivationValues[i] = nn.neuronVector[i - nn.nrOfParameters];
        }
        else if (nodeLayer == 3) {
            neuronActivationValues[i] = nn.outputVector[i - nn.nrOfHiddenNeurons];
        }
    }
    
    
}

/**
 * returns a vector of the nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
double ** findConnectedNeurons(struct NeuralNetwork nn, int neuron) {

    double ** connectedNodes = malloc(sizeof(int)*nn.neuronsPerLayer);

    int nodeLayer = getNeuronLayer(nn, neuron);

    if (nodeLayer == 1) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            connectedNodes[i] = & nn.neuronVector[i];
        }
    }

    else if (nodeLayer == 2) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            int offset = neuron - nn.nrOfParameters + nn.neuronsPerLayer;
            connectedNodes[i] = & nn.neuronVector[offset + i];
        }
    }

    else if (nodeLayer == 3) {
        for (int i = 0; i < nn.nrOfOutputs; ++i) {
            connectedNodes[i] = & nn.outputVector[i];
        }
    }

    return connectedNodes;
}

/**
 * returns a vector of the weights connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected weights of (node 0 is the top parameter node).
 * @return A vector of the connected weights. */
double ** findConnectedWeights(struct NeuralNetwork nn, int neuron) {
    
    double ** connectedWeights = malloc(sizeof(int)*(nn.neuronsPerLayer*nn.neuronsPerLayer));

    int nodeLayer = getNeuronLayer(nn, neuron);

    if (nodeLayer == 1) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            int offset = neuron * nn.neuronsPerLayer;
            connectedWeights[i] = & nn.weightMatrix[i + offset];
        }
    }

    else if (nodeLayer == 2) {
        int nodeLayer = (neuron-nn.nrOfParameters) / nn.neuronsPerLayer;
        int parameterLayerWeights = nn.nrOfParameters * nn.neuronsPerLayer;
        int offset = nodeLayer * nn.weightsPerLayer + parameterLayerWeights + neuron * nn.neuronsPerLayer;
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            connectedWeights[i] = & nn.weightMatrix[offset + i];
        }
    }

    else if (nodeLayer == 3) {
        int parameterLayerWeights = nn.nrOfParameters * nn.neuronsPerLayer;
        int hiddenLayerWeights = nn.weightsPerLayer * nn.nrOfHiddenLayers;
        int offset = parameterLayerWeights + hiddenLayerWeights + neuron * nn.nrOfOutputs;
        for (int i = 0; i < nn.nrOfOutputs; ++i) {
            connectedWeights[i] = & nn.weightMatrix[offset + i];
        }
    }

    return connectedWeights;
}