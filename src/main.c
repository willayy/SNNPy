#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"
#include "sigmoid.h"

int main() {
    /*
    struct NeuralNetwork nn = createNeuralNetwork(2, 2, 4, 2);

    double * input = (double *) malloc(sizeof(double)*2);
    input[0] = 0.5;
    input[1] = 0.5;

    double * desiredOutput = (double *) malloc(sizeof(double)*2);
    desiredOutput[0] = 0.6;
    desiredOutput[1] = 0.4;

    inputDataToNeuralNetwork(nn, input);
    */

   printf("Hello, %f\n", antiSigmoid(0.99919));
   printf("Hello, %f\n", antiSigmoid(0.99929));
   printf("Hello, %f\n", antiSigmoid(0.99939));
   printf("Hello, %f\n", antiSigmoid(0.99949));
   printf("Hello, %f\n", antiSigmoid(0.99959));
   printf("Hello, %f\n", antiSigmoid(0.99969));
   printf("Hello, %f\n", antiSigmoid(0.99979));
   printf("Hello, %f\n", antiSigmoid(0.99989));
   printf("Hello, %f\n", antiSigmoid(0.99910));
   printf("Hello, %f\n", antiSigmoid(0.99911));
   printf("Hello, %f\n", antiSigmoid(0.99912));
   printf("Hello, %f\n", antiSigmoid(0.99913));
    return 0;
}
