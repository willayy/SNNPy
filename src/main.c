#include <stdio.h>
#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "vectorOperations.h"

int main() {
    
    struct NeuralNetwork nn = createNeuralNetwork(2, 2, 5, 3);
    int a = 29;
    int b = 10;
    int c = a / b;
    printf("%d", c);

    return 0;
}
