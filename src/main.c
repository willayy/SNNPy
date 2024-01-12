#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 2, 3, 2);

    double * input1 = (double*) malloc(sizeof(double) * 2);
    input1[0] = 0;
    input1[1] = 1;
    double * desiredOutput1 = (double*) malloc(sizeof(double) * 2);
    desiredOutput1[0] = 1;
    desiredOutput1[1] = 0;

    double cost = 0;

    for (int i = 0; i < 10000; i++) {
        cost = trainOnData(nn, input1, desiredOutput1);
        if (i == 0 || i== 9999) {printf("Cost: %f \n", cost);}
    }

    printf("Output: %f %f \n", nn.outputVector[0], nn.outputVector[1]);


    return 0;
}
