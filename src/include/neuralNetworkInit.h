#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(NeuralNetwork * nn, int nrOfInputs, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

        void setInputLayerActivationFunction(NeuralNetwork * nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative);

        void setHiddenLayerActivationFunction(NeuralNetwork * nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative);

        void setOutputLayerActivationFunction(NeuralNetwork * nn, dblAdblR activationFunction, dblAdblR activationFunctionDerivative);

        void resetNeuralNetwork(NeuralNetwork * nn);

        void initWeightsXavierUniform(NeuralNetwork * nn);

        void initWeightsXavierNormal(NeuralNetwork * nn);

        void initBiasesConstant(NeuralNetwork * nn, double b);

        void initBiasesRandomUniform(NeuralNetwork * nn, double minb, double maxb);

        void initWeightsRandomUniform(NeuralNetwork * nn, double minw, double maxw);
#endif