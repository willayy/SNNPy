#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 2, 4, 2);
    double * input = (double *) malloc(sizeof(double)*2);
    input[0] = 0.5;
    input[1] = 0.5;
    double * desiredOutput = (double *) malloc(sizeof(double)*2);
    desiredOutput[0] = 0.6;
    desiredOutput[1] = 0.4;

    for (int i = 0; i < 100; i++) {
        
        double cost = trainOnData(nn, input, desiredOutput, 0.1
        , 0.2);
        inputDataToNeuralNetwork(nn, input);
        printf("Output: %f, %f Cost: %f Iteration: %i\n", nn.outputVector[0], nn.outputVector[1], cost, i);
    }

    return 0;
}
