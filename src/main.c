#include <stdio.h>
#include <stdlib.h>

#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"
#include "sigmoid.h"

int main() {

    struct NeuralNetwork nn = createNeuralNetwork(2, 3, 4, 2);

    double * input = (double *) malloc(sizeof(double)*2);
    input[0] = 1;
    input[1] = 0;

    double * desiredOutput = (double *) malloc(sizeof(double)*2);
    desiredOutput[0] = 0;
    desiredOutput[1] = 1;

    inputDataToNeuralNetwork(nn, input);

    for (int i = 0; i < 100; i++) {
        double cost = trainOnData(nn, input, desiredOutput, 0.4, 0.1);
        printf("Iteration: %d Cost: %f Output: %f, %f First layer gradients: %f, %f, %f, %f \n", i, cost, nn.outputVector[0], nn.outputVector[1], nn.hiddenVector[0], nn.hiddenVector[1], nn.hiddenVector[2], nn.hiddenVector[3]);
    }
    
    freeNeuralNetwork(nn);

    return 0;
}
