#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralNetworkStructs.h"
#include "neuralNetworkInit.h"
#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkUtility.h"
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
    setRngSeed(time(NULL));

    // Create neural network
    NeuralNetwork * nn  = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 4, 1, 4, 16);
    setInputLayerActivationFunction(nn, &linear, &linearDerivative);
    setHiddenLayerActivationFunction(nn, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative);
    setOutputLayerActivationFunction(nn, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.1);

    int epochs = 50000;
    int batchSize = 16;
    double epochCost;
    double lrw = 0.03;
    double lrb = 0.01;
    int indexes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    for (int i = 0; i < epochs; i++) {

        epochCost = 0;
        GradientBatch * gb = (GradientBatch *) malloc(sizeof(GradientBatch));
        initGradientBatch(gb, batchSize);
        fisherYatesShuffle(indexes, 16);

        for (int j = 0; j < batchSize; j++) {
            int inputIndex = indexes[j];
            double * output = inputDataToNeuralNetwork(nn, inputs[inputIndex]);
            epochCost += crossEntropyCostFunction(output, desOutputs[inputIndex], nn->nrOfOutputNeurons);
            gb->gradientVectors[j] = computeGradients(nn, desOutputs[inputIndex], &crossEntropyCostFunctionDerivative);
            free(output);
        }

        GradientVector * avgGradient = averageGradients(gb);
        
        optimize(nn, avgGradient, lrw, lrb);

        printf("Epoch %d, avg batch cost: %f\n", i, epochCost/batchSize);

        freeGradientVector(avgGradient);
        freeGradientBatch(gb);
    }

    // Printing the output of the neural network after its convergence / training
    for (int j = 0; j < 16; j++) {
        double * output = inputDataToNeuralNetwork(nn, inputs[j]);

        int biggestProbIndex = 0;
        for (int i = 0; i < 16; i++) {
            if (output[i] > output[biggestProbIndex]) {
                biggestProbIndex = i;
            }
        }  
        printf("Input: %.2f %.2f %.2f %.2f ", inputs[j][0], inputs[j][1], inputs[j][2], inputs[j][3]);
        printf("Output: %d, with %.3f probability,\n", biggestProbIndex, output[biggestProbIndex]);
        free(output);
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