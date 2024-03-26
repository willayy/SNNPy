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

    int epochs = 250000;
    int batchSize = 16;
    double epochCost;
    double lrw = 0.08;
    double lrb = 0.08;
    double lambda = 0;

    printf("Training setup for the neural network with %d epochs, the neural network should convert 4 bit binary numbers into integers ranging from 0-15\n", epochs);
    printf("Press any key to start training: ");  scanf("Press any key to continue...");
   
    trainNeuralNetworkOnBatch(nn, inputs, desOutputs, epochs, batchSize, lrw, lrb, NULL, NULL, lambda, crossEntropyCostFunctionDerivative, crossEntropyCostFunction, 1);

    // Printing the output of the neural network after its convergence / training
    for (int j = 0; j < 16; j++) {
        double * output = inputDataToNeuralNetwork(nn, inputs[j]);
        int biggestOutputIndex = findBiggestOutputIndex(output, nn->nrOfOutputNeurons);
        printf("Input: %.0f %.0f %.0f %.0f ", inputs[j][0], inputs[j][1], inputs[j][2], inputs[j][3]);
        printf("Output: %d, with %.3f activation,\n", biggestOutputIndex, output[biggestOutputIndex]);
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