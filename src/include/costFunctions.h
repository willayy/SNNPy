#include "neuralNetworkStructs.h"

#ifndef costFunctions_h
    #define costFunctions_h

        double sqrCostFunctionDerivative(double output, double desiredOutput);

        double sqrCostFunction(double * output, double * desiredOutput, int outputSize);

        double crossEntropyCostFunctionDerivative(double output, double desiredOutput);

        double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize);

        double l1Regularization(NeuralNetwork * nn);

        double l2Regularization(NeuralNetwork * nn);

        double l1RegularizationDerivative(double weight);

        double l2RegularizationDerivative(double weight);
#endif