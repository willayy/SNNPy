#include <stdio.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"

int main() {
    struct NeuralNetwork n = createNeuralNetwork(2, 1, 2, 1);
    struct ParameterLayer p = n.parameterLayer;
    struct NeuronLayer l = *(n.intermediateLayers);
    struct NeuronLayer o = n.outputLayer;
    double * output = vectorMatrixMul(p.parameters,l.edges, 2, 2); 
    printf("Output: %f, %f\n", output[0], output[1]);
    return 0;
}