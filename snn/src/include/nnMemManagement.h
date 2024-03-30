#include "neuralNetworkStructs.h"

#ifndef nnMemManagement_h
    #define nnMemManagement_h

    void freeDblPtr(double * ptr);

    void freeNeuralNetwork(NeuralNetwork * nn);

    void freeNeuron(Neuron * n);

    void freeNeuronGradient(NeuronGradient * ng);

    void freeGradientVector(GradientVector * gv);

    void freeGradientBatch(GradientBatch * gb);

#endif