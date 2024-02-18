#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "testing.h"
#include <stdlib.h>

int main() {

    struct NeuralNetwork * nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 1, 1, 1, 1);
    int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "numberOfConnectedNeurons 1, 1, 1, 1 network 1");
    int_assertEqual(numberOfConnectedNeurons(nn,1), 1, "numberOfConnectedNeurons 1, 1, 1, 1 network 2");
    freeNeuralNetwork(nn);

    nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 1, 3, 4, 1);
    int_assertEqual(numberOfConnectedNeurons(nn,0), 4, "numberOfConnectedNeurons 1, 3, 4, 1 network 1");
    int_assertEqual(numberOfConnectedNeurons(nn,1), 4, "numberOfConnectedNeurons 1, 3, 4, 1 network 2");
    int_assertEqual(numberOfConnectedNeurons(nn,2), 4, "numberOfConnectedNeurons 1, 3, 4, 1 network 3");

    return 0;
}