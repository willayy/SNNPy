#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h

        void trainNeuralNetworkOnBatch(NeuralNetwork * nn, double ** inputs, double ** labels, int epochs, int batchSize, 
                               double lrw, double lrb, double lambda, int verbose);
#endif