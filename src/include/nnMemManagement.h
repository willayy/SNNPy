#include "neuralNetworkStructs.h"

#ifndef nnMemManagement_h
    #define nnMemManagement_h

    void freeNeuralNetwork(NeuralNetwork * nn);

    void freeGradient(NeuronGradient * grad);

    void freeGradients(NeuronGradient ** batch, int nrOfGradients);

    void freeGradientsBatch(NeuronGradient *** batch, int nrOfGradients, int batchSize);

#endif