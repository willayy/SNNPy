#include "neuralNetworkStructs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

    double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb);

#endif