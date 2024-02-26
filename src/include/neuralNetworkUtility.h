#include "neuralNetworkStructs.h"

#ifndef neuronUtility_h
    #define neuronUtility_h

        int * findConnectedNeuronIndexes(struct NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronActivations(struct NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronValues(struct NeuralNetwork * nn, int neuron);

        double * findOutputWeights(struct NeuralNetwork * nn, int neuron);

        double * findConnectedNeuronBiases(struct NeuralNetwork * nn, int neuron);

        int isNeuronLastInLayer(struct NeuralNetwork * nn, int neuron);

        int isNeuronLastInLastlayer(struct NeuralNetwork * nn, int neuron);

        int numberOfConnectedNeurons(struct NeuralNetwork * nn, int neuron);
#endif