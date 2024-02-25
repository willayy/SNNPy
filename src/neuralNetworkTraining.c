#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

/**
 * Calculates the derivatives for the output layer of the neural network.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network. */
void outputLayerDerivatives(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA_intA costFunctionDerivative, int batchSize) {

    for (int i = nn->nrOfNeurons - 1; i >= (nn->nrOfNeurons - nn->nrOfOutputNeurons); i--) {
        
        double * neuronValue = findNeuronValue(nn, i);
        double * neuronActivation = findNeuronActivation(nn, i);
        int outputIndex = nn->nrOfNeurons - i - 1;
        double dCdA = costFunctionDerivative(nn->neuronActivationVector[i], desiredOutput[outputIndex], batchSize);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronValue[0]);
        neuronActivation[0] = dCdA * dAdZ;
    }
}

/**
 * Calculates the derivatives for the hidden layers of the neural network. 
 * @param nn The neural network. */
void hiddenLayerDerivatives(struct NeuralNetwork * nn) {

    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i >= nn->nrOfParameterNeurons; i--) {
        
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double * neuronValue = findNeuronValue(nn, i);
        double * connectedNeurons = findConnectedNeuronActivations(nn, i);
        double * connectedWeights = findConnectedWeights(nn, i);
        double derivativeSum = 0;

        for (int j = 0; j < nrOfConnectedNeurons; j++) {

            double dZdA = connectedWeights[j];
            derivativeSum += dZdA * connectedNeurons[j];
        }

        double dAdZ = nn->activationFunctionDerivative(neuronValue[0]);
        nn->neuronActivationVector[i] = derivativeSum * dAdZ; 
    }
}

void optimizeWeight(double * weight, double frontNeuronValue, double backNeuronValue, double lrw) {

    double dZdW = backNeuronValue; 
    weight[0] -= lrw * dZdW * frontNeuronValue; 
}

void optimizeBias(double * bias, double frontNeuronValue, double lrb) {

    double dZdB = 1;
    bias[0] -= lrb * dZdB * frontNeuronValue;
}

/**
 * Optimizes the weights and biases of the neural network.
 * @param nn The neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void optimize(struct NeuralNetwork * nn, double lrw, double lrb) {

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {

        double * backNeuronValue = findNeuronValue(nn, i);
        double * bias = findBias(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double * connectedNeurons = findConnectedNeuronActivations(nn, i);
        double * connectedWeights = findConnectedWeights(nn, i);
        double sumConnectedNeurons = 0;

        for (int j = 0; j < nrOfConnectedNeurons; j++) {

            double * weight = &connectedWeights[j];
            optimizeWeight(weight, connectedNeurons[j], backNeuronValue[0], lrw);
            sumConnectedNeurons += connectedNeurons[j];
        }

        optimizeBias(bias, sumConnectedNeurons, lrb);
    }
}

/**
 * Computes the gradient for the neural network.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param costFunctionDerivative The derivative of the cost function.
 * @param batchSize The size of the batch. */
double * computeGradient(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA_intA costFunctionDerivative, int batchSize) {

    outputLayerDerivatives(nn, desiredOutput, costFunctionDerivative, batchSize);
    
    hiddenLayerDerivatives(nn);

    return nn->neuronActivationVector;
}

/**
 * Fits the neural network to the desired output.
 * @param nn The neural network.
 * @param gradients The gradients of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void fit(struct NeuralNetwork * nn, double * gradients, double lrw, double lrb) {
    
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = gradients[i];
    }

    optimize(nn, lrw, lrb);
}
