#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(struct NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs, dblA activationFunction, dblA activationFunctionDerivative, dblA lastLayerActivationFunction, dblA lastLayerActivationFunctionDerivative);

        void resetNeuralNetwork(struct NeuralNetwork * nn);

        void freeNeuralNetwork(struct NeuralNetwork * nn);

        void initWeightsXavierUniform(struct NeuralNetwork * nn, unsigned int seed);

        void initWeightsXavierNormal(struct NeuralNetwork * nn, unsigned int seed);

        void initBiasesConstant(struct NeuralNetwork * nn, double b);

        void initBiasesRandomUniform(struct NeuralNetwork * nn, double * bRange, unsigned int seed);

        void initWeightsRandomUniform(struct NeuralNetwork * nn, double * wRange, unsigned int seed);
#endif