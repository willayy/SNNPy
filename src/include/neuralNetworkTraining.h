#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        NeuronGradient ** computeGradients(const NeuralNetwork * nn, double * partialGradients);

        double * computePartialGradients(const NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative);

        NeuronGradient ** averageGradients(const NeuralNetwork * nn, NeuronGradient *** sumNg, int batchSize, int nrOfNeurons);

        void optimize(NeuralNetwork * nn, NeuronGradient ** avgNg, double lrw, double lrb);
#endif