#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralNetworkStructs.h"
#include "neuralNetworkInit.h"
#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkUtility.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "randomValueGenerator.h"
#include "activationFunctions.h"
#include "nnMemManagement.h"

int main() {

    printf("\nRunning a simple neural network convergence test\n\n");

    int testSumConvergence = 0;

    // Create data set

    double ** inputs = (double **) malloc(sizeof(double *) * 16);
    double ** desOutputs = (double **) malloc(sizeof(double *) * 16);

    for (int i = 0; i < 16; i++) {
        inputs[i] = (double *) malloc(sizeof(double) * 4);
        desOutputs[i] = (double *) malloc(sizeof(double) * 16);
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            inputs[i][j] = (double) (i >> j & 1);
        }
        for (int j = 0; j < 16; j++) {
            desOutputs[i][j] = (double) (i == j);
        }
    }

    // Initialize random number generator
    setSeed(time(NULL));

    // Create neural network
    NeuralNetwork * nn  = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 4, 2, 4, 16);
    initNeuralNetworkFunctions(nn, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.1);

    int epochs = 100000;
    int batchSize = 16;
    double epochCost;
    double lrw = 0.03;
    double lrb = 0.01;

    for (int i = 0; i < epochs; i++) {
        epochCost = 0;
        GradientBatch * gb = (GradientBatch *) malloc(sizeof(GradientBatch));
        initGradientBatch(gb, batchSize);

        for (int j = 0; j < batchSize; j++) {
            double * output = inputDataToNeuralNetwork(nn, inputs[j]);
            epochCost += sqrCostFunction(output, desOutputs[j], nn->nrOfOutputNeurons);
            double * partialGradients = computePartialGradients(nn, desOutputs[j], &sqrCostFunctionDerivative);
            gb->gradientVectors[j] = computeGradients(nn, partialGradients);

            free(partialGradients);
            free(output);
        }

        GradientVector * avgGradient = averageGradients(gb);
        
        optimize(nn, avgGradient, lrw, lrb);

        printf("Epoch %d, cost: %f\n", i, epochCost/batchSize);

        freeGradientVector(avgGradient);
        freeGradientBatch(gb);
    }

    freeNeuralNetwork(nn);

    return testSumConvergence;
}