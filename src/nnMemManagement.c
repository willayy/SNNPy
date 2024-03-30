#include <stdlib.h>
#include "neuralNetworkStructs.h"

void freeNeuron(Neuron * n) {
    free(n->weights);
    free(n->connectedNeurons);
    free(n);
}

/**
 * Frees the memory allocated for a neural network.
 * @param nn: the neural network to free. */
void freeNeuralNetwork(NeuralNetwork * nn) {
    free(nn->inputLayerActivationFunctions);
    free(nn->hiddenLayerActivationFunctions);
    free(nn->outputLayerActivationFunctions);
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        freeNeuron(nn->neurons[i]);
    }
    free(nn->neurons);
    free(nn);
}

void freeNeuronGradient(NeuronGradient * ng) {
    free(ng->weightGradient);
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