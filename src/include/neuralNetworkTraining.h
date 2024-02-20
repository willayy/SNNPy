#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        double fit(struct NeuralNetwork * nn, double * desiredOutput, double lrw, double lrb, dblP_dblP_intA costFunction, dblA_dblA costFunctionDerivative);
#endif