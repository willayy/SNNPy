#include <stdlib.h>

#include "neuralNetworkOperations.h"

#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "vectorOperations.h"
#include "neuralNetworkStructs.h"
#include "activationFunctions.h"

void propogateForward(struct NeuralNetwork * nn, double * inputData) {

    // Set the input data to the parameter neurons.
    for (int i = 0; i < nn->nrOfParameterNeurons; i++) {

        nn->neuronValueVector[i] = inputData[i];
        nn->parameterVector[i] = nn->activationFunction(inputData[i]);
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {

        double * neuron = findNeuronActivation(nn, i);
        double * weigths =  findOutputWeights(nn, i);
        double * connectedNeurons =  findConnectedNeuronActivations(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double * result = vectorMulCopy(weigths, neuron[0], nrOfConnectedNeurons);
        vectorAdd(connectedNeurons, result, nrOfConnectedNeurons);
        free(result);

        // If the neuron is the last in the 
        if (isNeuronLastInLayer(nn, i)) {
            vectorExtend(nn->neuronValueVector, connectedNeurons, (i+1), nrOfConnectedNeurons);
            vectorAdd(nn->neuronActivationVector + i + 1, nn->biasVector + i + 1, nrOfConnectedNeurons);
            if (i == nn->nrOfNeurons - nn->nrOfOutputNeurons - 1) {
                vectorOperation(nn->neuronActivationVector + i + 1, nn->lastLayerActivationFunction, nrOfConnectedNeurons);
            } else {
                vectorOperation(nn->neuronActivationVector + i + 1, nn->activationFunction, nrOfConnectedNeurons);
            }
        }
    }
}

/**
 * Calculates the output of the neural network from a double array input.
 * @param nn The neural network to calculate the output of.
 * @param inputData The input data to the neural network. Must be of the same size as the input layer of the neural network.
 * @return The output of the network. */
double * inputDataToNeuralNetwork(struct NeuralNetwork * nn, double * inputData) {

    resetNeuralNetwork(nn); // reset the neural network.

    propogateForward(nn, inputData);

    double * result = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);

    for (int i = 0; i < nn->nrOfOutputNeurons; i++) {
        result[i] = nn->outputVector[i];
    }
    
    return result;

}