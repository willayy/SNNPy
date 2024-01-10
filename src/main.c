#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 4, 2, 2);

    double * input = (double*) malloc(sizeof(double) * 2);
    input[0] = 1;
    input[1] = 1;
    double * desiredOutput = (double*) malloc(sizeof(double) * 2);
    desiredOutput[0] = 0.5;
    desiredOutput[1] = 0.7;

    for (int i = 0; i < 10000; i++) {
        double * cost = trainOnData(nn, input, desiredOutput);
        free(cost);
    }

    inputDataToNeuralNetwork(nn, input);
    printf("Output: %f %f\n", nn.outputVector[0], nn.outputVector[1]);
    
    return 0;
}
