#include "neuralNetworkStructs.h"
#include "funcPtrs.h"

#ifndef neuralNetworkTraining_h
    #define neuralNetworkTraining_h
    
        double * computeGradient(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA_intA costFunctionDerivative, int batchSize);

        void fit(struct NeuralNetwork * nn, double * gradients, double lrw, double lrb);
#endif