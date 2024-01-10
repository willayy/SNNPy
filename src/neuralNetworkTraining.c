#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "sigmoid.h"
#include <stdlib.h>

static double * paramNullvector;
static double * outputNullVector;
static double * nullVector;
static double lrw = 0.50; // Learning rate weight
static double lrb = 0.10; // Learning rate bias


/**
 * Calculates the cost of the neural network on a given input and desired output. 
 * Calculating how good bad the current biases and edges are for a certain input.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double * costFunction(double * output, double * desiredOutput, int outputSize) {

    double *  cost = (double*) malloc(sizeof(double) * outputSize);

    for (int i = 0; i < outputSize; i++) {
        cost[i] = (desiredOutput[i]-output[i]);
    }

    return cost;
}

/**
 * Backpropagates the cost through the neural network.
 * @param nn The neural network.
 * @param cost The cost of the neural network on a given input and desired output. */
void backPropagateOutput(struct NeuralNetwork nn, double * cost) {

    vectorReplace(nn.outputVector, cost, nn.nrOfOutputs);

    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.neuronsPerLayer * nn.neuronsPerLayer * nn.nrOfLayers;
    double * tempBiases = nn.biasVector + nn.neuronsPerLayer * nn.nrOfLayers;
    
    for (int i = 0; i < nn.nrOfOutputs; i++) {
        double * deltaWeights = vectorMul(tempWeights, lrw * nn.outputVector[i], nn.neuronsPerLayer);
        double * deltaBias = vectorMul(tempBiases, lrb * nn.outputVector[i], nn.neuronsPerLayer);
        vectorAdd(tempWeights, deltaWeights, nn.neuronsPerLayer);
        vectorAdd(tempBiases, deltaBias, nn.neuronsPerLayer);
        tempWeights += nn.neuronsPerLayer;
    }

    tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.neuronsPerLayer * nn.neuronsPerLayer * nn.nrOfLayers;
    tempBiases = nn.biasVector + nn.neuronsPerLayer * (nn.nrOfLayers-1);
    double * tempNeurons = nn.neuronVector + nn.neuronsPerLayer * (nn.nrOfLayers-1);

    vectorReplace(tempNeurons, outputNullVector, nn.neuronsPerLayer);

    for (int i = 0; i < nn.nrOfOutputs; i++) {
        double * temp = vectorMul(tempWeights, nn.outputVector[i], nn.neuronsPerLayer);
        vectorAdd(tempNeurons, temp, nn.neuronsPerLayer);
        free(temp);
    }

    vectorAdd(tempNeurons, tempBiases, nn.neuronsPerLayer);

    vectorOperation(tempNeurons, sigmoid, nn.neuronsPerLayer);

}

/**
 * Backpropagates the cost through the hidden layers of the neural network.
 * @param nn The neural network. */
void backPropogateHiddenlayers(struct NeuralNetwork nn) {

    double * startNeurons = nn.neuronVector + nn.neuronsPerLayer * (nn.nrOfLayers - 1);
    double * tempWeights = nn.weightMatrix + nn.nrOfParameters * nn.neuronsPerLayer + nn.neuronsPerLayer * nn.neuronsPerLayer * (nn.nrOfLayers - 1);
    double * tempBiases = nn.biasVector + nn.neuronsPerLayer * (nn.nrOfLayers - 2);
    double * tempNeurons = nn.neuronVector + nn.neuronsPerLayer * (nn.nrOfLayers - 2);

    for (int i = nn.nrOfLayers - 1; i > 0; i--) {

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            double * deltaWeights = vectorMul(tempWeights, lrw * startNeurons[j], nn.neuronsPerLayer);
            double * deltaBias = vectorMul(tempBiases, lrb * startNeurons[j], nn.neuronsPerLayer);
            vectorAdd(tempWeights + j * nn.neuronsPerLayer, deltaWeights, nn.neuronsPerLayer);
            vectorAdd(tempBiases, deltaBias, nn.neuronsPerLayer);
            free(deltaWeights);
            free(deltaBias);
        }

        vectorReplace(tempNeurons, nullVector, nn.neuronsPerLayer);

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            double * temp = vectorMul(tempWeights + j * nn.neuronsPerLayer, startNeurons[j], nn.neuronsPerLayer);
            vectorAdd(tempNeurons, temp, nn.neuronsPerLayer);
            free(temp);
        }

        vectorAdd(tempNeurons, tempBiases, nn.neuronsPerLayer);

        vectorOperation(tempNeurons, sigmoid, nn.neuronsPerLayer);
        
        startNeurons -= nn.neuronsPerLayer;
        tempWeights -= nn.neuronsPerLayer * nn.neuronsPerLayer;
        tempBiases -= nn.neuronsPerLayer;
        tempNeurons -= nn.neuronsPerLayer;

    }

}

void backPropogateInput(struct NeuralNetwork nn) {
    
    double * tempWeights = nn.weightMatrix;
    double * tempBiases = nn.biasVector;
    double * tempNeurons = nn.neuronVector;

    for (int i = 0; i < nn.neuronsPerLayer; i++) {
        double * deltaWeights = vectorMul(tempWeights, lrw * nn.neuronVector[i], nn.nrOfParameters);
        double * deltaBias = vectorMul(tempBiases, lrb * nn.neuronVector[i], nn.nrOfParameters);
        vectorAdd(tempWeights, deltaWeights, nn.nrOfParameters);
        vectorAdd(tempBiases, deltaBias, nn.nrOfParameters);
        free(deltaWeights);
        free(deltaBias);
    }

    vectorReplace(tempNeurons, paramNullvector, nn.nrOfParameters);

    for (int i = 0; i < nn.neuronsPerLayer; i++) {
        double * temp = vectorMul(tempWeights, nn.neuronVector[i], nn.nrOfParameters);
        vectorAdd(tempNeurons, temp, nn.nrOfParameters);
        free(temp);
    }

    vectorAdd(tempNeurons, tempBiases, nn.nrOfParameters);

    vectorOperation(tempNeurons, sigmoid, nn.nrOfParameters);

}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input of the neural network.
 * @param desiredOutput The desired output of the neural network. */
double * trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput) {

    paramNullvector = (double*) malloc(sizeof(double) * nn.nrOfParameters);
    outputNullVector = (double*) malloc(sizeof(double) * nn.nrOfOutputs);
    nullVector = (double*) malloc(sizeof(double) * nn.neuronsPerLayer);

    for (int i = 0; i < nn.nrOfParameters; i++) {paramNullvector[i] = 0;}
    for (int i = 0; i < nn.nrOfOutputs; i++) {outputNullVector[i] = 0;}
    for (int i = 0; i < nn.neuronsPerLayer; i++) {nullVector[i] = 0;}
    
    inputDataToNeuralNetwork(nn, input);
    
    double * cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
    
    backPropagateOutput(nn, cost);

    backPropogateHiddenlayers(nn);

    backPropogateInput(nn);
    
    return cost;
}
