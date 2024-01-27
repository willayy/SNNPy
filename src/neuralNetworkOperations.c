#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkInit.h"
#include "sigmoid.h"
#include <stdlib.h>

void propogateForwardParams(struct NeuralNetwork nn, double * inputData) {

    nn.parameterVector = inputData;
    
    vectorOperation(nn.parameterVector, sigmoid, nn.nrOfParameters);

    for (int i = 0; i < nn.neuronsPerLayer; i++) {
        nn.neuronVector[i] = dotProduct(nn.parameterVector, nn.weightMatrix + i * nn.nrOfParameters, nn.nrOfParameters);
    }

    vectorAdd(nn.neuronVector, nn.biasVector, nn.neuronsPerLayer);

    vectorOperation(nn.neuronVector, sigmoid, nn.neuronsPerLayer);
 
}

void propogateForwardHiddenLayers(struct NeuralNetwork nn) {

    for (int i = 0; i < nn.nrOfHiddenLayers; i++) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            double * neuronVector = nn.neuronVector + i * nn.neuronsPerLayer;
            double * weightVector = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + i * nn.neuronsPerLayer*nn.neuronsPerLayer + j * nn.neuronsPerLayer;
            nn.neuronVector[(i+1)*nn.neuronsPerLayer + j] = dotProduct(neuronVector, weightVector, nn.neuronsPerLayer);
        }

        vectorAdd(nn.neuronVector + (i+1)*nn.neuronsPerLayer, nn.biasVector + (i+1)*nn.neuronsPerLayer, nn.neuronsPerLayer);

        vectorOperation(nn.neuronVector + (i+1)*nn.neuronsPerLayer, sigmoid, nn.neuronsPerLayer);
    }
}

void propogateForwardOutput(struct NeuralNetwork nn) {

    for (int i = 0; i < nn.nrOfOutputs; i++) {
        double * neuronVector = nn.neuronVector + nn.nrOfHiddenLayers * nn.neuronsPerLayer;
        double * weightVector = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.nrOfHiddenLayers * nn.neuronsPerLayer * nn.neuronsPerLayer + i * nn.neuronsPerLayer;
        nn.outputVector[i] = dotProduct(neuronVector, weightVector, nn.neuronsPerLayer);
    }
    
    vectorOperation(nn.outputVector, sigmoid, nn.nrOfOutputs);
}

/**
 * Calculates the output of the neural network from a double array input.
 * @param nn The neural network to calculate the output of.
 * @param inputData The input data to the neural network. Must be of the same size as the input layer of the neural network. */
void inputDataToNeuralNetwork(struct NeuralNetwork nn, double * inputData) {

    propogateForwardParams(nn, inputData);

    propogateForwardHiddenLayers(nn);

    propogateForwardOutput(nn);

}