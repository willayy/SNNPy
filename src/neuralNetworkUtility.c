#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include <stdlib.h>



/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons) {
        return nn->nrOfOutputNeurons;
    }
    
    return 0;

}

/**
 * Returns a vector of the indexes of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
int * findConnectedNeuronIndexes(struct NeuralNetwork * nn, int neuron) {

    int * connectedNeurons;

    if (neuron < nn->nrOfParameterNeurons) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->neuronsPerLayer);
        for (int i = 0; i < nn->neuronsPerLayer; i++) {
            connectedNeurons[i] = i + (nn->nrOfParameterNeurons);
        }
        return connectedNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->neuronsPerLayer);
        for (int i = 0; i < nn->neuronsPerLayer; i++) {
            connectedNeurons[i] = i + (nn->nrOfParameterNeurons) + (1 + (neuron / nn->neuronsPerLayer)) * nn->neuronsPerLayer;
        }
        return connectedNeurons;
    }
    else if (neuron < nn->nrOfNeurons) {
        connectedNeurons = (int *) malloc(sizeof(int) * nn->nrOfOutputNeurons);
        for (int i = 0; i < nn->nrOfOutputNeurons; i++) {
            connectedNeurons[i] = i + (nn->nrOfParameterNeurons) + nn->nrOfHiddenNeurons;
        }
        return connectedNeurons;
    }

    return 0;

}

/**
 * Returns a vector of the activations of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected activations of (node 0 is the top parameter node).
 * @return A vector of the connected activations. */
double * findConnectedNeuronActivations(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronActivationVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->hiddenActivationVector + (1 + (neuron / nn->neuronsPerLayer)) * nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfNeurons) {
        return nn->activationOutputVector;
    }

    return 0;
}

/**
 * Returns a vector of the values of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected values of (node 0 is the top parameter node).
 * @return A vector of the connected values. */
double * findConnectedNeuronValues(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronValueVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->hiddenValueVector + nn->nrOfParameterNeurons + (1 + (neuron / nn->neuronsPerLayer)) * nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfNeurons) {
        return nn->outputValueVector;
    }

    return 0;
}

/**
 * Returns a vector of the biases of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected biases of (node 0 is the top parameter node).
 * @return A vector of the connected biases. */
double * findConnectedNeuronBiases(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->biasVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->biasVector + (1 + (neuron / nn->neuronsPerLayer)) * nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfNeurons) {
        return nn->biasVector + nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons;
    }

    return 0;
}

/**
 * returns a vector of the weights connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected weights of (node 0 is the top parameter node).
 * @return A vector of the connected weights. */
double * findOutputWeights(struct NeuralNetwork * nn, int neuron) {
    if (neuron >= 0 && neuron < nn->nrOfNeurons) {
        return (nn->weightMatrix[neuron]);
    }

    return 0;

}

/**
 * Is the given neuron the last in its layer.
 * @param nn The neural network.
 * @param neuron The neuron to check.
 * @return 1 if the neuron is the last in its layer, 0 otherwise. */
int isNeuronLastInLayer(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons && neuron == nn->nrOfParameterNeurons - 1) {
        return 1;
    }

    else if ((neuron >= nn->nrOfParameterNeurons)
        && (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons)
        && ((neuron - nn->nrOfParameterNeurons + 1) % nn->neuronsPerLayer == 0)) {
            return 1;
        }

    if (neuron == nn->nrOfNeurons - 1) {
        return 1;
    }

    return 0;
}

/**
 * Is the given neuron the last neuron in the final hidden layer of the network.
 * @param nn The neural network.
 * @param neuron The neuron to check.
 * @return 1 if the neuron is the last in the final hidden layer, 0 otherwise. */
int isNeuronLastInLastlayer(struct NeuralNetwork * nn, int neuron) {

    if (neuron == nn->nrOfNeurons - nn->nrOfOutputNeurons - 1) {
        return 1;
    }

    return 0;
}