#include "neuralNetworkStructs.h"
#include <stdlib.h>

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(NeuralNetwork * nn) {

    free(nn->neuronActivationVector);

    free(nn->biasVector);

    free(nn->neuronValueVector);

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        free(nn->weightMatrix[i]);
    }
    
    free(nn->weightMatrix);

    free(nn);
}

/**
 * frees the memory allocated for a neuron gradient.
 * @param ng: the neuron gradient to free. */
void freeGradient(NeuronGradient * grad) {
    free(grad->weightGradient);
    free(grad->biasGradient);
    free(grad);
}


void freeGradients(NeuronGradient ** gradArray, int nrOfGradients) {
    for (int i = 0; i < nrOfGradients; i++) {
        freeGradient(gradArray[i]);
    }
    free(gradArray);
}

void freeGradientsBatch(NeuronGradient *** batch, int nrOfGradients, int batchSize) {
    for (int i = 0; i < batchSize; i++) {
        freeGradients(batch[i], nrOfGradients);
    }
    free(batch);
}