#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkTraining.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkInit.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkOperations.h"
#include "nnMemManagement.h"

int main() {
    
    NeuralNetwork * nn = createNeuralNetworkPtr();
    initNeuralNetwork(nn, 2, 1, 2, 2);
    setInputLayerActivationFunction(nn, &linear, &linearDerivative);
    setHiddenLayerActivationFunction(nn, &sigmoid, &sigmoidDerivative);
    setOutputLayerActivationFunction(nn, &sigmoid, &sigmoidDerivative);
    setCostFunction(nn, &sqrCostFunction, &sqrCostFunctionDerivative);
    setRegularization(nn, &noRegularization, &noRegularizationDerivative);
    initWeightsXavierNormal(nn);
    initBiasesConstant(nn, 0.1);

    double ** input = (double **) malloc(sizeof(double *) * 1);
    double ** labels = (double **) malloc(sizeof(double *) * 1);
    input[0] = (double *) malloc(sizeof(double) * 2);
    labels[0] = (double *) malloc(sizeof(double) * 2);
    input[0][0] = 1; input[0][1] = 0;
    labels[0][0] = 0; labels[0][1] = 1;

    trainNeuralNetworkOnBatch(nn, input, labels, 20000, 1, 0.08, 0.01, 0, 1);

    freeNeuralNetwork(nn);
}