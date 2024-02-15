#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "testing.h"

int main() {

    struct NeuralNetwork nn = createNeuralNetwork(1, 1, 1, 1);
    int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "numberOfConnectedNeurons mini network 1");
    int_assertEqual(numberOfConnectedNeurons(nn,1), 1, "numberOfConnectedNeurons mini network 2");
    freeNeuralNetwork(nn);

    


    return 0;
}