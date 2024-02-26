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

        nn->parameterValueVector[i] = inputData[i];
        nn->parameterValueVector[i] = nn->biasVector[i];
        nn->activationParameterVector[i] = nn->activationFunction(nn->neuronValueVector[i]);
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {

        double neuronActivation = nn->neuronActivationVector[i];
        double * connectedWeights = findOutputWeights(nn, i);
        double * ConnectedNeuronValues = findConnectedNeuronValues(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double * forwardPropogatedValues = vectorMulCopy(connectedWeights, neuronActivation, nrOfConnectedNeurons);
        vectorAdd(ConnectedNeuronValues, forwardPropogatedValues, nrOfConnectedNeurons);
        free(forwardPropogatedValues);

        if (isNeuronLastInLayer(nn, i)) {

            double * connectedActivationValues = findConnectedNeuronActivations(nn, i);
            vectorAdd(ConnectedNeuronValues, findConnectedNeuronBiases(nn, i), nrOfConnectedNeurons);
            vectorReplace(connectedActivationValues, ConnectedNeuronValues, nrOfConnectedNeurons);

            if (isNeuronLastInLastlayer(nn, i)) {
                vectorOperation(connectedActivationValues, nn->lastLayerActivationFunction, nrOfConnectedNeurons); 
            } else { 
                vectorOperation(connectedActivationValues, nn->activationFunction, nrOfConnectedNeurons); 
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
        result[i] = nn->activationOutputVector[i];
    }
    
    return result;

}