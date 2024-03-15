#include "neuralNetworkStructs.h"

#ifndef nnMemManagement_h
    #define nnMemManagement_h

    void freeNeuralNetwork(NeuralNetwork * nn);

    void freeNeuronGradient(NeuronGradient * ng);

    void freeGradientVector(GradientVector * gv);

    void freeGradientBatch(GradientBatch * gb);

#endif