#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "neuralNetworkOperations.h"
#include "sigmoid.h"
#include <stdlib.h>

void propogateForwardParams(struct NeuralNetwork nn, double * inputData) {

    nn.parameterVector = inputData;
    double * tempWeights = nn.weightMatrix;

    for (int i = 0; i < nn.nrOfParameters; i++) {

        double * temp = vectorMul(tempWeights, nn.parameterVector[i], nn.nrOfParameters);

        vectorAdd(nn.neuronVector, temp, nn.neuronsPerLayer);
        
        free(temp);

        tempWeights += nn.neuronsPerLayer;
    }

    vectorAdd(nn.neuronVector, nn.biasVector, nn.neuronsPerLayer);

    for (int i = 0; i < nn.neuronsPerLayer; i++) {
        nn.neuronVector[i] = sigmoid(nn.neuronVector[i]);
    }
 
}

void propogateForwardHiddenLayers(struct NeuralNetwork nn) {

    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer; // Skip the first layer as it has already been calculated.
    double * tempNeurons = nn.neuronVector;
    double * tempBias = nn.biasVector + nn.neuronsPerLayer;

    for (int i = 0; i < nn.nrOfLayers - 1; i++) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {

            double * temp = vectorMul(tempWeights, tempNeurons[j], nn.neuronsPerLayer);

            vectorAdd(tempNeurons + nn.neuronsPerLayer, temp, nn.neuronsPerLayer);

            tempWeights += nn.neuronsPerLayer;

            free(temp);
        }

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            tempNeurons[j + nn.neuronsPerLayer] += tempBias[j];
            tempNeurons[j + nn.neuronsPerLayer] = sigmoid(tempNeurons[j]);
        }

        tempBias += nn.neuronsPerLayer;
        tempNeurons += nn.neuronsPerLayer;
    }
}

void propogateForwardOutput(struct NeuralNetwork nn) {

    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.neuronsPerLayer * nn.neuronsPerLayer * nn.nrOfLayers;
    double * tempNeurons = nn.neuronVector + nn.neuronsPerLayer * (nn.nrOfLayers - 1);

    for (int i = 0; i < nn.nrOfOutputs; i++) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {

            double * temp = vectorMul(tempWeights, tempNeurons[j], nn.neuronsPerLayer);

            vectorAdd(nn.outputVector, temp, nn.nrOfOutputs);

            tempWeights += nn.neuronsPerLayer;

            free(temp);
        }

        nn.outputVector[i] = sigmoid(nn.outputVector[i]);

    }
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