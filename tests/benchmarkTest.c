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

int main() {

    printf("\nRunning a simple neural network convergence test\n\n");

    int testSumConvergence = 1;

    // Create data set

    double ** inputs = (double **) malloc(sizeof(double *) * 16);
    double ** outputs = (double **) malloc(sizeof(double *) * 16);

    for (int i = 0; i < 16; i++) {
        inputs[i] = (double *) malloc(sizeof(double) * 4);
        outputs[i] = (double *) malloc(sizeof(double) * 16);
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            inputs[i][j] = (double) (i >> j & 1);
        }
        for (int j = 0; j < 16; j++) {
            outputs[i][j] = (double) (i == j);
        }
    }

    // Create neural network

    setSeed(time(NULL));
    struct NeuralNetwork * nn  = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 4, 1, 4, 16, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.1);

    double epochCostSum;
    double * result;
    double * partialGradient;
    double ** sumBiasGradients = (double **) malloc(sizeof(double *) * 16);
    double *** sumWeightGradients = (double ***) malloc(sizeof(double **) * 16);
    double ** avgWeightGradients;
    double * avgBiasGradients;

    // train neural network
    
    for (int i = 0; i < 100; i++) {

        epochCostSum = 0;

        for (int j = 0; j < 16; j++) {
            result = inputDataToNeuralNetwork(nn, inputs[j]);
            epochCostSum += sqrCostFunction(result, outputs[j], 16);
            partialGradient = computePartialGradient(nn, outputs[j], &sqrCostFunctionDerivative);
            sumBiasGradients[j] = computeBiasGradients(nn, partialGradient);
            sumWeightGradients[j] = computeWeightGradients(nn, partialGradient);
            free(result);
            free(partialGradient);
        }

        avgWeightGradients = averageWeightGradients(nn, sumWeightGradients, 16);
        avgBiasGradients = averageBiasGradients(nn, sumBiasGradients, 16);

        optimize(nn, avgWeightGradients, avgBiasGradients, 0.01, 0.01);

        printf("Epoch %d, cost: %f\n", i, epochCostSum/16);

        // free all gradients in list of gradients
        for (int j = 0; j < 16; j++) {
            freeWeightGradients(sumWeightGradients[j], nn->nrOfNeurons);
            free(sumBiasGradients[j]);
        }

        // free top level pointer        
        free(sumBiasGradients);
        free(sumWeightGradients);

        // free averages
        freeWeightGradients(avgWeightGradients, nn->nrOfNeurons);
        free(avgBiasGradients);
        
    }

    freeNeuralNetwork(nn);

    for (int i = 0; i < 16; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);

    return testSumConvergence;
}