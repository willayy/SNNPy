#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        NeuronGradient ** computeGradients(NeuralNetwork * nn, double * partialGradients);

        double * computePartialGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative);

        NeuronGradient ** averageGradients(NeuralNetwork * nn, NeuronGradient *** sumNg, int batchSize);

        void optimize(NeuralNetwork * nn, NeuronGradient ** avgNg, double lrw, double lrb);
#endif