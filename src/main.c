#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "vectorOperations.h"

int main() {
    struct NeuralNetwork nn = createNeuralNetwork(2, 20, 10, 2);

    double inputData[2] = {-23, 54};

    double * output = inputDataToNeuralNetwork(nn, inputData);

    printf("Output: %f\n", output[0]);
    printf("Output: %f\n", output[1]);

    return 0;
}