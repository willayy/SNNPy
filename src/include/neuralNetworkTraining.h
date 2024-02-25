#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void freeWeightGradients(double ** gradients, int nrOfNeurons);

        double ** averageWeightGradients(struct NeuralNetwork * nn, double *** sumGradients, double batchSize);

        double * averageBiasGradients(struct NeuralNetwork * nn, double ** sumGradients, double batchSize);

        double ** computeWeightGradients(struct NeuralNetwork * nn, double * partialGradients);

        double * computeBiasGradients(struct NeuralNetwork * nn, double * partialGradients);

        double * computePartialGradient(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA costFunctionDerivative);

        void optimize(struct NeuralNetwork * nn, double ** Wgrad, double * Bgrad, double lrw, double lrb);
#endif