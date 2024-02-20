#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralNetworkStructs.h"
#include "neuralNetworkInit.h"
#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"

int main() {

    printf("\nRunning a simple neural network convergence test\n\n");

    int testSumConvergence = 0;

    struct NeuralNetwork * nn  = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 4, 2, 16, 16, &sigmoid, &sigmoidDerivative);
    initWeightsXavierNormal(nn, time(NULL));
    initBiasesConstant(nn, 0.1);

    double * input0 = (double *) malloc(sizeof(double) * 4);
    double * input1 = (double *) malloc(sizeof(double) * 4);
    double * input2 = (double *) malloc(sizeof(double) * 4);
    double * input3 = (double *) malloc(sizeof(double) * 4);
    double * input4 = (double *) malloc(sizeof(double) * 4);
    double * input5 = (double *) malloc(sizeof(double) * 4);
    double * input6 = (double *) malloc(sizeof(double) * 4);
    double * input7 = (double *) malloc(sizeof(double) * 4);
    double * input8 = (double *) malloc(sizeof(double) * 4);
    double * input9 = (double *) malloc(sizeof(double) * 4);
    double * input10 = (double *) malloc(sizeof(double) * 4);
    double * input11 = (double *) malloc(sizeof(double) * 4);
    double * input12 = (double *) malloc(sizeof(double) * 4);
    double * input13 = (double *) malloc(sizeof(double) * 4);
    double * input14 = (double *) malloc(sizeof(double) * 4);
    double * input15 = (double *) malloc(sizeof(double) * 4);

    double * output0 = (double *) malloc(sizeof(double) * 16);
    double * output1 = (double *) malloc(sizeof(double) * 16);
    double * output2 = (double *) malloc(sizeof(double) * 16);
    double * output3 = (double *) malloc(sizeof(double) * 16);
    double * output4 = (double *) malloc(sizeof(double) * 16);
    double * output5 = (double *) malloc(sizeof(double) * 16);
    double * output6 = (double *) malloc(sizeof(double) * 16);
    double * output7 = (double *) malloc(sizeof(double) * 16);
    double * output8 = (double *) malloc(sizeof(double) * 16);
    double * output9 = (double *) malloc(sizeof(double) * 16);
    double * output10 = (double *) malloc(sizeof(double) * 16);
    double * output11 = (double *) malloc(sizeof(double) * 16);
    double * output12 = (double *) malloc(sizeof(double) * 16);
    double * output13 = (double *) malloc(sizeof(double) * 16);
    double * output14 = (double *) malloc(sizeof(double) * 16);
    double * output15 = (double *) malloc(sizeof(double) * 16);    

    input0 = (double[]) {0, 0, 0, 0};
    input1 = (double[]) {1, 0, 0, 0};
    input2 = (double[]) {0, 1, 0, 0};
    input3 = (double[]) {1, 1, 0, 0};
    input4 = (double[]) {0, 0, 1, 0};
    input5 = (double[]) {1, 0, 1, 0};
    input6 = (double[]) {0, 1, 1, 0};
    input7 = (double[]) {1, 1, 1, 0};
    input8 = (double[]) {0, 0, 0, 1};
    input9 = (double[]) {1, 0, 0, 1};
    input10 = (double[]) {0, 1, 0, 1};
    input11 = (double[]) {1, 1, 0, 1};
    input12 = (double[]) {0, 0, 1, 1};
    input13 = (double[]) {1, 0, 1, 1};
    input14 = (double[]) {0, 1, 1, 1};
    input15 = (double[]) {1, 1, 1, 1};

    output0 = (double[]) {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    output1 = (double[]) {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    output2 = (double[]) {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0};
    output3 = (double[]) {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
    output4 = (double[]) {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    output5 = (double[]) {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0};
    output6 = (double[]) {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0};
    output7 = (double[]) {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
    output8 = (double[]) {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0};
    output9 = (double[]) {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0};
    output10 = (double[]) {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0};
    output11 = (double[]) {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0};
    output12 = (double[]) {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0};
    output13 = (double[]) {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0};
    output14 = (double[]) {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0};
    output15 = (double[]) {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};

    double cost;

    for (int i = 0; i < 10000; i++) {
        cost = 0;
        inputDataToNeuralNetwork(nn, input0);
        cost += fit(nn, output0, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input1);
        cost +=fit(nn, output1, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input2);
        cost +=fit(nn, output2, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input3);
        cost +=fit(nn, output3, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input4);
        cost +=fit(nn, output4, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input5);
        cost +=fit(nn, output5, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input6);
        cost +=fit(nn, output6, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input7);
        cost +=fit(nn, output7, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input8);
        cost +=fit(nn, output8, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input9);
        cost +=fit(nn, output9, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input10);
        cost +=fit(nn, output10, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input11);
        cost +=fit(nn, output11, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input12);
        cost +=fit(nn, output12, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input13);
        cost +=fit(nn, output13, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input14);
        cost +=fit(nn, output14, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);
        inputDataToNeuralNetwork(nn, input15);
        cost +=fit(nn, output15, 0.1, 0.01, &sqrCostFunction, &sqrCostFunctionDerivative);

        if (cost < 0.1) {
            printf("Converged after %d iterations\n", i);
            break;
        }
    }
    printf("Neural network did not converge final Cost: %f\n", cost);
}