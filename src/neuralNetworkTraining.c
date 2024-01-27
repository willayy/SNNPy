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
        double dCdA = costFunctionDerivative(nn.outputVector[i], desiredOutput[i]);
        double dAdZ = sigmoidDerivative(nn.outputVector[i]);
        nn.outputVector[i] = dCdA * dAdZ;
    }
}

void hiddenLayerDerivatives(struct NeuralNetwork nn) {

}

void optimizeWeight(double * weight, double * frontNeuron, double backNeuronValue, double lrw) {
    double dZdW = backNeuronValue;
    weight[0] -= lrw * dZdW * frontNeuron[0]; 
}

void backPropogateNeuron(struct NeuralNetwork nn, int neuronIndex, double neuronActivationValues, double lrw, double lrb) {
    double ** connectedNodes = findConnectedNeurons(nn, neuronIndex);
    double ** connectedWeights = findConnectedWeights(nn, neuronIndex);
    int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, neuronIndex);
}

/**
 * Backpropogates the error through the neural network and optimizes the weights and biases.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void backPropogate(struct NeuralNetwork nn, double * desiredOutput, double lrw, double lrb) {

    double * neuronActivationValues = getNeuronActivationValues(nn);

    outputLayerDerivatives(nn, desiredOutput);

    hiddenLayerDerivatives(nn);

    int nodesToBackPropagate = nn.nrOfHiddenNodes + nn.nrOfParameters;

    for (int i = nodesToBackPropagate; i >= 0; i--) {
        backPropogateNeuron(nn, i, neuronActivationValues[i], lrw, lrb);
    }
    
}

/**
 * Trains the neural network on a given input and desired output.
 * @param nn The neural network.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network. */
double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb, double delta) {
    
    inputDataToNeuralNetwork(nn, input);
    
    double cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputs);
    
    backPropogate(nn, desiredOutput, lrw, lrb);

    return cost;
}
