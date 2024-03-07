#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void freeMatrix(double ** matrix, int rows);

        double ** averageWeightGradients(NeuralNetwork * nn, double *** sumGradients, double batchSize);

        double * averageBiasGradients(NeuralNetwork * nn, double ** sumGradients, double batchSize);

        double ** computeWeightGradients(NeuralNetwork * nn, double * partialGradients);

        double * computeBiasGradients(NeuralNetwork * nn, double * partialGradients);

        double * computePartialGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative);

        void optimize(NeuralNetwork * nn, double ** Wgrad, double * Bgrad, double lrw, double lrb);
#endif