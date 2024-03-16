#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include <stdlib.h>



/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(NeuralNetwork * nn, int neuron) {

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
int * findConnectedNeuronIndexes(NeuralNetwork * nn, int neuron) {

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
            connectedNeurons[i] = i + (nn->nrOfParameterNeurons) + (neuron / nn->neuronsPerLayer) * nn->neuronsPerLayer;
        }
        return connectedNeurons;
    }
    
    else if (neuron < nn->nrOfNeurons - nn->nrOfOutputNeurons) {
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
double * findConnectedNeuronActivations(NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronActivationVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->hiddenActivationVector + (neuron / nn->neuronsPerLayer) * nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfNeurons - nn->nrOfOutputNeurons) {
        return nn->activationOutputVector;
    }

    return 0;
}

/**
 * Returns a vector of the values of the connected neurons in the forward direction.
 * @param nn The neural network.
 * @param neuron The node to find the connected values of (node 0 is the top parameter node).
 * @return A vector of the connected values. */
double * findConnectedNeuronValues(NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronValueVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->hiddenValueVector + nn->nrOfParameterNeurons + (neuron / nn->neuronsPerLayer) * nn->neuronsPerLayer;
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
double * findConnectedNeuronBiases(NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->biasVector;
    }
    else if (neuron <= nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->biasVector + (neuron / nn->neuronsPerLayer) * nn->neuronsPerLayer;
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
double * findOutputWeights(NeuralNetwork * nn, int neuron) {
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
int isNeuronLastInLayer(NeuralNetwork * nn, int neuron) {

    if (neuron == nn->nrOfParameterNeurons - 1) {
        return 1;
    }

    else if ((neuron >= nn->nrOfParameterNeurons) && (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons)
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
int isNeuronLastInHiddenlayer(NeuralNetwork * nn, int neuron) {

    if (neuron == nn->nrOfNeurons - nn->nrOfOutputNeurons - 1) {
        return 1;
    }

    return 0;
}

/**
 * Finds the bias of a given neuron.
 * @param nn The neural network.
 * @param neuron The neuron to find the bias of.
 * @return The bias of the neuron. */
double findBias(NeuralNetwork * nn, int neuron) {
    if (neuron < nn->nrOfParameterNeurons || neuron >= nn->nrOfNeurons) {
        return 0;
    }
    return nn->biasVector[neuron - nn->nrOfParameterNeurons];
}

/**
 * Finds the activation of a given neuron.
 * @param nn The neural network.
 * @param neuron The neuron to find the activation of.
 * @return The activation of the neuron. */
double findActivation(NeuralNetwork * nn, int neuron) {
    if (neuron < nn->nrOfParameterNeurons || neuron >= nn->nrOfNeurons) {
        return 0;
    }
    return nn->neuronActivationVector[neuron - nn->nrOfParameterNeurons];
}