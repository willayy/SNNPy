#include <stdlib.h>
#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"

/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfInputNeurons) {
        return nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfInputNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfInputNeurons + nn->nrOfHiddenNeurons) {
        return nn->nrOfOutputNeurons;
    }
    
    return 0;

}

/**
 * Returns a vector of the indexes of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
int * findConnectedNeuronIndexes(NeuralNetwork * nn, int neuron) {

    int * connectedNeurons;

    if (neuron < nn->nrOfInputNeurons) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->neuronsPerLayer);
        for (int i = 0; i < nn->neuronsPerLayer; i++) {
            connectedNeurons[i] = i + (nn->nrOfInputNeurons);
        }
        return connectedNeurons;
    }
    
    else if (neuron < nn->nrOfInputNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->neuronsPerLayer);
        for (int i = 0; i < nn->neuronsPerLayer; i++) {
            connectedNeurons[i] = i + (nn->nrOfInputNeurons) + (neuron / nn->neuronsPerLayer) * nn->neuronsPerLayer;
        }
        return connectedNeurons;
    }
    
    else if (neuron < nn->nrOfNeurons - nn->nrOfOutputNeurons) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->nrOfOutputNeurons);
        for (int i = 0; i < nn->nrOfOutputNeurons; i++) {
            connectedNeurons[i] = i + (nn->nrOfInputNeurons) + nn->nrOfHiddenNeurons;
        }
        return connectedNeurons;
    }

    return 0;

}

/**
 * Returns the index of the biggest activation value of the output layer.
 * @param output The output of the neural network.
 * @return The index of the biggest activation value. */
int findBiggestOutputIndex(double * output, int outputSize) {
    int biggestProbIndex = 0;
        for (int i = 0; i < outputSize; i++) {
            if (output[i] > output[biggestProbIndex]) {
                biggestProbIndex = i;
            }
        }  
    return biggestProbIndex;
}