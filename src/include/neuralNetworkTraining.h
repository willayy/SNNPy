#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void fit(struct NeuralNetwork * nn, double * avgOutput, double * desiredOutput, double lrw, double lrb, dblA_dblA costFunctionDerivative);
#endif