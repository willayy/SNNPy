#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include <stdlib.h>



/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the number of connected nodes of (node 0 is the top parameter node).
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
 * Returns pointer to a neurons activartion value in the neural network.
 * @param nn The neural network.
 * @param node The neuron. */
double * findNeuronActivation(struct NeuralNetwork * nn, int neuron) {
    
    if (neuron >= 0 && neuron < nn->nrOfNeurons) {
        return & (nn->neuronActivationVector[neuron]);
    }

    return 0;

}

/**
 * Returns pointer to a neurons value in the neural network.
 * @param nn The neural network.
 * @param node The neuron. */
double * findNeuronValue(struct NeuralNetwork * nn, int neuron) {
    if (neuron >= 0 && neuron < nn->nrOfNeurons) {

        return & (nn->neuronValueVector[neuron]);
    }

    return 0;

}

/**
 * Returns pointer to a bias in the neural network.
 * @param nn The neural network.
 * @param node The neuron. */
double * findBias(struct NeuralNetwork * nn, int neuron) {
    if (neuron >= 0 && neuron < nn->nrOfNeurons) {
        return & (nn->biasVector[neuron]);
    }

    return 0;

}

/**
 * returns a vector of the nodes activation values connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
double * findConnectedNeuronActivations(struct NeuralNetwork * nn, int neuron) {

    if (neuron < nn->nrOfParameterNeurons) {
        return nn->neuronActivationVector + nn->nrOfParameterNeurons;
    }
    else if (neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons - nn->neuronsPerLayer) {
        return nn->hiddenVector + (1 + (neuron / nn->neuronsPerLayer)) * nn->neuronsPerLayer;
    }
    else if (neuron < nn->nrOfNeurons) {
        return nn->outputVector;
    }

    return 0;

}

/**
 * returns a vector of the weights connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected weights of (node 0 is the top parameter node).
 * @return A vector of the connected weights. */
double * findConnectedWeights(struct NeuralNetwork * nn, int neuron) {
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

    else if (neuron >= nn->nrOfParameterNeurons && neuron < nn->nrOfParameterNeurons + nn->nrOfHiddenNeurons) {
        if ((neuron - nn->nrOfParameterNeurons + 1) % nn->neuronsPerLayer == 0) {
            return 1;
        }
    }

    if (neuron == nn->nrOfNeurons - 1) {
        return 1;
    }

    return 0;
}
