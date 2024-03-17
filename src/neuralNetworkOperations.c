#include <stdlib.h>

#include "neuralNetworkOperations.h"

#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "vectorOperations.h"
#include "neuralNetworkStructs.h"
#include "activationFunctions.h"

void propogateForward(NeuralNetwork * nn, double * inputData) {

    // Set the input layer to the input data.
    for (int i = 0; i < nn->nrOfInputNeurons; i++) {
        nn->neurons[i]->Z = inputData[i];
    }

    int nrOfConnections;
    Neuron * n;

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        n = nn->neurons[i];
        n->Z += n->bias;
        n->A = n->activationFunctions[0](n->Z);
        nrOfConnections = n->conections;
        for (int j = 0; j < nrOfConnections; j++) {
            n->connectedNeurons[j]->Z += n->weights[j] * n->A;
        }
    }
}

/**
 * Calculates the output of the neural network from a double array input.
 * @param nn The neural network to calculate the output of.
 * @param inputData The input data to the neural network. Must be of the same size as the input layer of the neural network.
 * @return The output of the network. */
double * inputDataToNeuralNetwork(NeuralNetwork * nn, double * inputData) {

    resetNeuralNetwork(nn); // reset the neural network.

    propogateForward(nn, inputData);

    double * result = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);

    for (int i = 0; i < nn->nrOfOutputNeurons; i++) {
        result[i] = nn->outputLayer[i]->A;
    }
    
    return result;

}