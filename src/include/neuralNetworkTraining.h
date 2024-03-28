#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void initGradientBatch(GradientBatch * gb, int batchSize); 

        GradientVector * computeGradients(NeuralNetwork * nn, double * desiredOutput, dblA_dbLA_dblR costFunctionDerivative);

        GradientVector * averageGradients(GradientBatch * gb);

        void optimize(NeuralNetwork * nn, GradientVector * avgNg, dblA_dblR regularizationDerivative, double lrw, double lrb, double lambda);

        void trainNeuralNetworkOnBatch(NeuralNetwork * nn, double ** inputs, double ** labels, int epochs, int batchSize, 
                               double lrw, double lrb, double lambda, int verbose);
#endif