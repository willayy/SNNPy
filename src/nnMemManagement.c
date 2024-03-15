#include "neuralNetworkStructs.h"
#include <stdlib.h>

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(NeuralNetwork * nn) {

    free(nn->neuronActivationVector);

    free(nn->biasVector);

    free(nn->neuronValueVector);

    for (int i = 0; i < nn->nrOfNeurons-nn->nrOfOutputNeurons; i++) {
        free(nn->weightMatrix[i]);
    }
    
    free(nn->weightMatrix);

    free(nn);
}

void freeNeuronGradient(NeuronGradient * ng) {
    free(ng->weightGradient);
    free(ng->biasGradient);
    free(ng);
}

void freeGradientVector(GradientVector * gv) {
    for (int i = 0; i < gv->nrOfNeurons; i++) {
        freeNeuronGradient(gv->gradients[i]);
    }
    free(gv->gradients);
    free(gv);
}

void freeGradientBatch(GradientBatch * gb) {
    for (int i = 0; i < gb->batchSize; i++) {
        freeGradientVector(gb->gradientVectors[i]);
    }
    free(gb->gradientVectors);
    free(gb);
}