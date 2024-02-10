#include <stdio.h>
#include <stdlib.h>

#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"
#include "sigmoid.h"

int main() {

    struct NeuralNetwork nn = createNeuralNetwork(2, 1, 2, 2);

    double * input = (double *) malloc(sizeof(double)*2);
    input[0] = 1;
    input[1] = 0.1;
    

    double * desiredOutput = (double *) malloc(sizeof(double)*2);
    desiredOutput[0] = 0.1;
    desiredOutput[1] = 1;
    

    for (int i = 0; i < 10000; i++) {
        resetNeuralNetwork(nn);
        inputDataToNeuralNetwork(nn, input);
        double cost = fit(nn, desiredOutput, 0.01, 0.2);
        printf("Iteration: %d Cost: %f \n", i, cost);
    }
    
    free(input);
    free(desiredOutput);
    freeNeuralNetwork(nn);

    return 0;
}
