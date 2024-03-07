#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(NeuralNetwork * nn, int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

        void initNeuralNetworkFunctions(NeuralNetwork * nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative, dblAdblR lastLayerActivationFunction, dblAdblR lastLayerActivationFunctionDerivative);

        void resetNeuralNetwork(NeuralNetwork * nn);

        void freeNeuralNetwork(NeuralNetwork * nn);

        void initWeightsXavierUniform(NeuralNetwork * nn);

        void initWeightsXavierNormal(NeuralNetwork * nn);

        void initBiasesConstant(NeuralNetwork * nn, double b);

        void initBiasesRandomUniform(NeuralNetwork * nn, double * bRange);

        void initWeightsRandomUniform(NeuralNetwork * nn, double * wRange);
#endif