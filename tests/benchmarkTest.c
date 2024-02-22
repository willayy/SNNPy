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
#include "activationFunctions.h"

int main() {

    printf("\nRunning a simple neural network convergence test\n\n");

    int testSumConvergence = 0;

    struct NeuralNetwork * nn  = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 4, 1, 4, 1, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative, &linear, &linearDerivative);
    initWeightsXavierNormal(nn, 2);
    initBiasesConstant(nn, 0.1);

    double ** inputs = (double **) malloc(sizeof(double *) * 16);
    double ** outputs = (double **) malloc(sizeof(double *) * 16);

    for (int i = 0; i < 16; i++) {
        inputs[i] = (double *) malloc(sizeof(double) * 4);
        outputs[i] = (double *) malloc(sizeof(double) * 1);
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 4; j++) {
            inputs[i][j] = (double) (i >> j & 1);
        }
        outputs[i][0] = (double) (i);
    }

    double epochCostSum;
    
    for (int i = 0; i < 10; i++) {

        epochCostSum = 0;

        for (int j = 0; j < 16; j++) {
            double * result = inputDataToNeuralNetwork(nn, inputs[j]);
            epochCostSum += sqrCostFunction(result, outputs[j], 1);
            double * weights = findConnectedWeights(nn, 0);
            fit(nn, result, outputs[j], 0.1, 0.01, &sqrCostFunctionDerivative);
            free(result);
        }

        printf("Epoch %d, cost: %f\n", i, epochCostSum/16);

    }

    freeNeuralNetwork(nn);

    return testSumConvergence;
}