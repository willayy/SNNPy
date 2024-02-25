#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(struct NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs, dblA activationFunction, dblA activationFunctionDerivative, dblA lastLayerActivationFunction, dblA lastLayerActivationFunctionDerivative);

        void resetNeuralNetwork(struct NeuralNetwork * nn);

        void freeNeuralNetwork(struct NeuralNetwork * nn);

        void initWeightsXavierUniform(struct NeuralNetwork * nn);

        void initWeightsXavierNormal(struct NeuralNetwork * nn);

        void initBiasesConstant(struct NeuralNetwork * nn, double b);

        void initBiasesRandomUniform(struct NeuralNetwork * nn, double * bRange);

        void initWeightsRandomUniform(struct NeuralNetwork * nn, double * wRange);
#endif