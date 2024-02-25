#include "neuralNetworkStructs.h"

#ifndef neuronUtility_h
    #define neuronUtility_h

        double * findConnectedNeuronActivations(struct NeuralNetwork * nn, int neuron);

        double * findOutputWeights(struct NeuralNetwork * nn, int neuron);

        double * findNeuronActivation(struct NeuralNetwork * nn, int neuron);

        double * findNeuronValue(struct NeuralNetwork * nn, int neuron);

        double * findBias(struct NeuralNetwork * nn, int neuron);

        int isNeuronLastInLayer(struct NeuralNetwork * nn, int neuron);

        int numberOfConnectedNeurons(struct NeuralNetwork * nn, int neuron);
#endif