#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

    double ** findConnectedNeurons(struct NeuralNetwork nn, int neuron);

    double ** findConnectedWeights(struct NeuralNetwork nn, int neuron);

    double * getOutputNeuronActivationValues(struct NeuralNetwork nn);

    double * getNeuronActiviationValues(struct NeuralNetwork nn);

    double * findNeuron(struct NeuralNetwork nn, int neuron);

    int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron);

#endif