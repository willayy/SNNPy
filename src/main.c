#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 4, 2, 2);
    double * input = (double*) malloc(sizeof(double) * 2);
    input[0] = 0;
    input[1] = 0;
    inputDataToNeuralNetwork(nn, input);
    
    printf("%f\n", nn.outputVector[0]);
    printf("%f\n", nn.outputVector[1]);
    return 0;
}
