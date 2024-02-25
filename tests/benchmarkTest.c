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

    setSeed(time(NULL));

    struct NeuralNetwork * nn  = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 4, 1, 4, 16, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.1);

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

    double epochCostSum;
    double * result;
    double ** weightGradients;
    double const * biasGradients;
    double * partialGradient;
    double * sumBiasGradients;
    double ** sumWeightGradients;

    
    for (int i = 0; i < 100; i++) {

        epochCostSum = 0;

        for (int j = 0; j < 16; j++) {
            result = inputDataToNeuralNetwork(nn, inputs[j]);
            epochCostSum += sqrCostFunction(result, outputs[j], 16);
            partialGradient = computePartialGradient(nn, result, outputs[j], &sqrCostFunctionDerivative);
            weightGradients = computeGradientsWeights(nn, partialGradient, 16);
            biasGradients = computeGradientsBiases(nn, partialGradient, 16);
            vectorAdd(sumBiasGradients, biasGradients, nn->nrOfNeurons);
            for (int k = 0; k < nn->nrOfNeurons; k++) {
                vectorAdd(sumWeightGradients[k], weightGradients[k], nn->nrOfNeurons);
            }
            free(result);
            free(partialGradient);
            for (int k = 0; k < nn->nrOfNeurons; k++) { free(weightGradients[k]); }
        }

        optimize(nn, sumWeightGradients, sumBiasGradients, 0.01, 0.01);

        printf("Epoch %d, cost: %f\n", i, epochCostSum/16);

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