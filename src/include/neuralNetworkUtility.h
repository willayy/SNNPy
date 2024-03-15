#include "neuralNetworkStructs.h"

#ifndef neuronUtility_h
    #define neuronUtility_h

        int * findConnectedNeuronIndexes(NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronActivations(NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronValues(NeuralNetwork * nn, int neuron);

        double * findOutputWeights(NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronBiases(NeuralNetwork * nn, int neuron);

        int isNeuronLastInLayer(NeuralNetwork * nn, int neuron);

        int isNeuronLastInHiddenlayer(NeuralNetwork * nn, int neuron);

        int numberOfConnectedNeurons(NeuralNetwork * nn, int neuron);
#endif