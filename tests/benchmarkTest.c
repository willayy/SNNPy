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
    initNeuralNetwork(nn, 4, 1, 4, 16);
    initNeuralNetworkFunctions(nn, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.20);

    int epochs = 20000;
    int batchSize = 16;
    double epochCost;
    double lrw = 0.002;
    double lrb = 0.001;

    for (int i = 0; i < epochs; i++) {
        epochCost = 0;
        GradientBatch * gb = (GradientBatch *) malloc(sizeof(GradientBatch));
        initGradientBatch(gb, batchSize);

        if (i == 5000) {
            printf("Sumsing");
        }

        for (int j = 0; j < batchSize; j++) {
            double * output = inputDataToNeuralNetwork(nn, inputs[j]);
            epochCost += crossEntropyCostFunction(output, desOutputs[j], nn->nrOfOutputNeurons);
            gb->gradientVectors[j] = computeGradients(nn, desOutputs[j], &crossEntropyCostFunctionDerivative);
            free(output);
        }

        GradientVector * avgGradient = averageGradients(gb);
        
        optimize(nn, avgGradient, lrw, lrb);

        printf("Epoch %d, cost: %f\n", i, epochCost/batchSize);

        if (epochCost/batchSize < 1) {
            testSumConvergence = 1;
            printf("Converged after %d epochs\n", i);
            for (int j = 0; j < 16; j++) {
                double * output = inputDataToNeuralNetwork(nn, inputs[j]);
                printf("Input: %f %f %f %f ", inputs[j][0], inputs[j][1], inputs[j][2], inputs[j][3]);
                printf("Output: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,\n", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9], output[10], output[11], output[12], output[13], output[14], output[15]);
                free(output);
            }
            break;
        }

        freeGradientVector(avgGradient);
        freeGradientBatch(gb);
    }

    freeNeuralNetwork(nn);

    for (int i = 0; i < 16; i++) {
        free(inputs[i]);
        free(desOutputs[i]);
    }
    free(inputs);
    free(desOutputs);

    return testSumConvergence;
}