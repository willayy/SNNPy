#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

    double ** findConnectedNeurons(struct NeuralNetwork nn, int neuron);

    double ** findConnectedWeights(struct NeuralNetwork nn, int neuron);

    double * getNeuronActivationValues(struct NeuralNetwork nn);

    int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron);

#endif