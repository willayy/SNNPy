#include "neuralNetworkStructs.h"

#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

        void initNeuralNetwork(NeuralNetwork * nn, int nrOfInputs, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

        void setInputLayerActivationFunction(NeuralNetwork * nn, dblA_dblR activationFunction, dblA_dblR activationFunctionDerivative);

        void setHiddenLayerActivationFunction(NeuralNetwork * nn, dblA_dblR activationFunction, dblA_dblR activationFunctionDerivative);

        void setOutputLayerActivationFunction(NeuralNetwork * nn, dblA_dblR activationFunction, dblA_dblR activationFunctionDerivative);

        void resetNeuralNetwork(NeuralNetwork * nn);

        void initWeightsXavierUniform(NeuralNetwork * nn);

        void initWeightsXavierNormal(NeuralNetwork * nn);

        void initBiasesConstant(NeuralNetwork * nn, double b);

        void initBiasesRandomUniform(NeuralNetwork * nn, double minb, double maxb);

        void initWeightsRandomUniform(NeuralNetwork * nn, double minw, double maxw);
#endif