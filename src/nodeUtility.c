#include "neuralNetworkStructs.h"
#include "nodeUtility.h"
#include <stdlib.h>

/**
 * returns a vector of the nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected nodes of.
 * @return A vector of the connected nodes. */
int * findConnectedNodes(struct NeuralNetwork nn, int node) {

    int * connectedNodes = malloc(sizeof(int)*nn.neuronsPerLayer);
    int isParamNode = node <= nn.nrOfParameters;
    int isHiddenNode = node > nn.nrOfParameters && node <= (nn.nrOfHiddenNodes - nn.neuronsPerLayer);
    int isHiddenToOutputNode = node > nn.nrOfHiddenNodes - nn.neuronsPerLayer && node <= nn.nrOfHiddenNodes;

    if (isParamNode) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            connectedNodes[i] = nn.neuronVector[i];
        }
    }

    else if (isHiddenNode) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            int offset = node - nn.nrOfParameters + nn.neuronsPerLayer;
            connectedNodes[i] = nn.neuronVector[offset + i];
        }
    }

    else if (isHiddenToOutputNode) {
        for (int i = 0; i < nn.nrOfOutputs; ++i) {
            connectedNodes[i] = nn.outputVector[i];
        }
    }
}

int * findConnectedWeights(struct NeuralNetwork nn, int node) {
    
}