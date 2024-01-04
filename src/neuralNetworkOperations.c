#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "neuralNetworkOperations.h"
#include "sigmoid.h"
#include <stdlib.h>

void propogateForwardParams(struct NeuralNetwork nn, double * inputData) {

    nn.parameterVector = inputData;

    for (int i = 0; i < nn.nrOfParameters; i++) {

        double * temp = vectorMul(nn.weightMatrix, nn.parameterVector[i], nn.nrOfParameters);

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            nn.neuronVector[j] += temp[j];
        }

        free(temp);
    }

    for (int i = 0; i < nn.neuronsPerLayer; i++) {
        nn.neuronVector[i] = sigmoid(nn.neuronVector[i]);
    }
}

void propogateForwardHiddenLayers(struct NeuralNetwork nn) {

    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer; // Skip the first layer as it has already been calculated.
    double * tempNeurons = nn.neuronVector; // Skip the first layer as it has already been calculated.

    for (int i = 0; i < nn.nrOfLayers - 1; i++) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {

            double * temp = vectorMul(tempWeights, tempNeurons[j], nn.neuronsPerLayer);

            for (int k = 0; k < nn.neuronsPerLayer; k++) {
                tempNeurons[k + nn.neuronsPerLayer] += temp[k];
            }

            tempWeights += nn.neuronsPerLayer;

            free(temp);
        }

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            tempNeurons[j + nn.neuronsPerLayer] = sigmoid(tempNeurons[j]);
        }

        tempNeurons += nn.neuronsPerLayer;
    }
}

void propogateForwardOutput(struct NeuralNetwork nn) {

    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.neuronsPerLayer * nn.neuronsPerLayer * nn.nrOfLayers;
    double * tempNeurons = nn.neuronVector + nn.neuronsPerLayer * (nn.nrOfLayers - 1);

    for (int i = 0; i < nn.nrOfOutputs; i++) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {

            double * temp = vectorMul(tempWeights, tempNeurons[j], nn.neuronsPerLayer);

            for (int k = 0; k < nn.nrOfOutputs; k++) {
                nn.outputVector[k] += temp[k];
            }

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