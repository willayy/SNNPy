#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(struct NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

        void resetNeuralNetwork(struct NeuralNetwork * nn);

        void freeNeuralNetwork(struct NeuralNetwork * nn);
#endif