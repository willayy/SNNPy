#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void initGradientBatch(GradientBatch * gb, int batchSize); 

        GradientVector * computeGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative);

        GradientVector * averageGradients(GradientBatch * gb);

        void optimize(NeuralNetwork * nn, GradientVector * avgNg, dblAdblR regularizationDerivative, double lrw, double lrb, double lambda);

        void trainNeuralNetworkOnBatch(NeuralNetwork * nn, double ** inputs, double ** labels, int epochs, 
                                       int batchSize, double lrw, double lrb, dblAdblR regularizationDerivative, nnAvoidR regularization, 
                                       double lambda, dblAdbLAdblR costFunctionDerivative);
#endif