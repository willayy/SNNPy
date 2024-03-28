#include "neuralNetworkStructs.h"

#ifndef costFunctions_h
    #define costFunctions_h

        double sqrCostFunctionDerivative(double output, double desiredOutput);

        double sqrCostFunction(double * output, double * desiredOutput, int outputSize);

        double crossEntropyCostFunctionDerivative(double output, double desiredOutput);

        double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize);

        double noRegularization(Neuron ** nv, int nrOfNeurons);

        double l1Regularization(Neuron ** nv, int nrOfNeurons);

        double l2Regularization(Neuron ** nv, int nrOfNeurons);

        double noRegularizationDerivative(double weight);

        double l1RegularizationDerivative(double weight);

        double l2RegularizationDerivative(double weight);
#endif