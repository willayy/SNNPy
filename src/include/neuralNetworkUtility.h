#include "neuralNetworkStructs.h"

#ifndef neuronUtility_h
    #define neuronUtility_h

        int * findConnectedNeuronIndexes(NeuralNetwork * nn, int neuron);

        int numberOfConnectedNeurons(NeuralNetwork * nn, int neuron);

        int findBiggestOutputIndex(double * output, int outputSize);
#endif