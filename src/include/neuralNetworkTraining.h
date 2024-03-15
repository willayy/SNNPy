#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void initGradientBatch(GradientBatch * gb, int batchSize); 

        GradientVector * computeGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative);

        GradientVector * averageGradients(GradientBatch * gb);

        void optimize(NeuralNetwork * nn, GradientVector * avgNg, double lrw, double lrb);
#endif