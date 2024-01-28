#include "nodeUtility.h"
#include "neuralNetworkTraining.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunction.h"
#include "sigmoid.h"
#include <stdlib.h>


void outputLayerDerivatives(struct NeuralNetwork nn, double * desiredOutput) {
    for (int i = 0; i < nn.nrOfOutputs; i++) {
        double * neuron = findNeuron(nn, i);
        double dCdA = costFunctionDerivative(nn.outputVector[i], desiredOutput[i]);
        double dAdZ = sigmoidDerivative(neuron[i]);
        nn.outputVector[i] = dCdA * dAdZ;
    }
}

void hiddenLayerDerivatives(struct NeuralNetwork nn) {
    for (int i = nn.nrOfHiddenNeurons + nn.nrOfParameters - 1; i > nn.nrOfParameters; i--) {

        double ** connectedNeurons = findConnectedNeurons(nn, i);
        double ** connectedWeights = findConnectedWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double * neuron = findNeuron(nn, i);
        double derivativeSum = 0;

        double dAdZ = sigmoidDerivative(neuron[0]);

        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdA = connectedWeights[j][0];
            double frontNeuron = connectedNeurons[j][0];
            derivativeSum += dZdA * dAdZ * frontNeuron;
        }

        free(connectedNeurons);
        free(connectedWeights);

        double avgDerivative = derivativeSum / nrOfConnectedNeurons;

        neuron[0] = derivativeSum;

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

void backPropogateNeuron(struct NeuralNetwork nn, int neuronIndex, double neuronActivationValue, double lrw, double lrb) {

    double ** connectedWeights = findConnectedWeights(nn, neuronIndex);
    double ** connectedNeurons = findConnectedNeurons(nn, neuronIndex);
    int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, neuronIndex);

    for (int i = 0; i < nrOfConnectedNeurons; i++) {
        double * weight = connectedWeights[i];
        double frontNeuronValue = connectedNeurons[i][0];
        optimizeWeight(weight, frontNeuronValue, neuronActivationValue, lrw);
    }

    if (neuronIndex >= nn.nrOfParameters) {
        double * bias = nn.biasVector + (neuronIndex - nn.nrOfParameters);
        optimizeBias(bias, neuronActivationValue, lrb);
    }

    free(connectedWeights);
    free(connectedNeurons);
}

/**
 * Backpropogates the error through the neural network and optimizes the weights and biases.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void backPropogate(struct NeuralNetwork nn, double * desiredOutput, double lrw, double lrb) {

    double * neuronActivationValues = getNeuronActiviationValues(nn); //Save the neuron activation values for later use.

    outputLayerDerivatives(nn, desiredOutput); //Calculate the derivatives for the output layer. Stored in the neural network output vector.  Replacing the activation values.

    hiddenLayerDerivatives(nn); //Calculate the derivatives for the hidden layers. Stored in the neural network neuron vector. Replacing the activation values.

    int nodesToBackPropagate = nn.nrOfHiddenNeurons + nn.nrOfParameters - 1;

    for (int i = nodesToBackPropagate; i >= 0; i--) {
        backPropogateNeuron(nn, i, neuronActivationValues[i], lrw, lrb);
    }
    
    /*
        free(outputNeuronActivationValues);
        free(neuronActivationValues);
    */
}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network. */
double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb) {
    
    inputDataToNeuralNetwork(nn, input);
    
    double cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
    
    backPropogate(nn, desiredOutput, lrw, lrb);

    return cost;
}
