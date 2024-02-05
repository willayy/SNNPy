#include "neuralNetworkStructs.h"

#ifndef neuronUtility_h
    #define neuronUtility_h

        double * findConnectedNeurons(struct NeuralNetwork nn, int neuron);

        double * findConnectedWeights(struct NeuralNetwork nn, int neuron);

        double * getActivationValues(struct NeuralNetwork nn);

        double * findNeuron(struct NeuralNetwork nn, int neuron);

        double * findBias(struct NeuralNetwork nn, int neuron);

        int isNeuronLastInLayer(struct NeuralNetwork nn, int neuron);

        int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron);
#endif