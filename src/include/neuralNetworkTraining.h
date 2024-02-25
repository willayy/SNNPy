#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        double ** computeGradientsWeights(struct NeuralNetwork * nn, const double * partialGradients, double batchSize);

        double * computeGradientsBiases(struct NeuralNetwork * nn, const double * partialGradients, double batchSize);

        double * computePartialGradient(struct NeuralNetwork * nn, const double * result, const double * desiredOutput, dblA_dblA costFunctionDerivative);

        void optimize(struct NeuralNetwork * nn, const double ** Wgrad, const double * Bgrad, double lrw, double lrb);
#endif