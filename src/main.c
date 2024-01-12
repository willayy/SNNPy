#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 2, 5, 3);

    double * input1 = (double*) malloc(sizeof(double) * 2);
    input1[0] = 0.3;
    input1[1] = 0.3;
    input1[2] = 0.3;
    double * desiredOutput1 = (double*) malloc(sizeof(double) * 2);
    desiredOutput1[0] = 0;
    desiredOutput1[1] = 0.9;
    desiredOutput1[2] = 0;

    double cost = 0;

    for (int i = 0; i < 20000; i++) {
        cost = trainOnData(nn, input1, desiredOutput1, 0.5, 0.1, 0.01);
        if (i == 0 || i== 19999) {printf("Cost : %f \n\n", cost);}
    }

    printf("Output: %f, %f, %f \n\n", nn.outputVector[0], nn.outputVector[1], nn.outputVector[2]);

    for (int i = 0; i < nn.nrOfWeights; i++) {
        printf("Weight: %f \n", nn.weightMatrix[i]);
    }

    return 0;
}
