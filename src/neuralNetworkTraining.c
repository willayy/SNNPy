#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunction.h"
#include "sigmoid.h"
#include "neuronUtility.h"

void optimizeWeight(double * weight, double frontNeuronValue, double backNeuronValue, double lrw) {
    double dZdW = backNeuronValue; // BUG ALERT BUG ALERT : BACK NEURON VALUE IS THE DERIVATIVE NOT THE VALUE OF THE NEURON
                                   // NEED TO SAVE THE VALUE OF THE NEURON IN THE PROPAGATE FORWARD FUNCTION OR IN THE DERIVATIVE FINDING FUNCTIONS
    weight[0] -= lrw * dZdW * frontNeuronValue; 
}

void optimizeBias(double * bias, double frontNeuronValue, double lrb) {
    double dZdB = 1;
    bias[0] -= lrb * dZdB * frontNeuronValue;
}

/**
 * Calculates the derivatives for the output layer of the neural network.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network. */
void outputLayerDerivatives(struct NeuralNetwork nn, double * desiredOutput) {
    for (int i = nn.nrOfNeurons - 1; i < (nn.nrOfNeurons - nn.nrOfOutputNeurons); i--) {
        double * neuron = findNeuron(nn, i);
        double dCdA = costFunctionDerivative(nn.outputVector[i], desiredOutput[i]);
        double dAdZ = sigmoidDerivative(antiSigmoid(neuron[0]));
        nn.outputVector[i] = dCdA * dAdZ;
    }
}

/**
 * Calculates the derivatives for the hidden layers of the neural network. 
 * @param nn The neural network. */
void hiddenLayerDerivatives(struct NeuralNetwork nn) {
    for (int i = nn.nrOfNeurons - nn.nrOfOutputNeurons - 1; i > nn.nrOfParameterNeurons; i--) {
        double * neuron = findNeuron(nn, i);
        double * connectedNeurons = findConnectedNeurons(nn, i);
        double * connectedWeights = findConnectedWeights(nn, i);
        double derivativeSum = 0;

        for (int j = 0; j < numberOfConnectedNeurons(nn, i); j++) {
            double dZdA = connectedWeights[j];
            derivativeSum += connectedNeurons[j] * dZdA;
        }

        double dAdZ = sigmoidDerivative(antiSigmoid(neuron[0]));
        nn.neuronVector[i] = derivativeSum * dAdZ; 
    }
}

/**
 * Optimizes the weights and biases of the neural network.
 * @param nn The neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void optimize(struct NeuralNetwork nn, double lrw, double lrb) {
    for (int i = 0; i < nn.nrOfNeurons - nn.nrOfOutputNeurons; i++) {
        double * neuron = findNeuron(nn, i);
        double * bias = findBias(nn, i);
        double * connectedNeurons = findConnectedNeurons(nn, i);
        double * connectedWeights = findConnectedWeights(nn, i);
        double sumConnectedNeurons = 0;

        for (int j = 0; j < numberOfConnectedNeurons(nn, i); j++) {
            optimizeWeight(&connectedWeights[j], connectedNeurons[j], neuron[0], lrw);
            sumConnectedNeurons += connectedNeurons[j];
        }

        optimizeBias(bias, sumConnectedNeurons, lrb);
    }
}

/**
 * Backpropogates the error through the neural network and optimizes the weights and biases.
 * @param nn The neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void backPropogate(struct NeuralNetwork nn, double * desiredOutput, double lrw, double lrb) {

    outputLayerDerivatives(nn, desiredOutput);
    
    hiddenLayerDerivatives(nn);

    optimize(nn, lrw, lrb);
    
}

/**
 * Trains the neural network on a given input and desired output.
 * This is done by first propogating the data forward through the neural network, then backpropogating the error and optimizing the weights and biases.
 * @param nn The neural network.
 * @param input The input vector of the neural network.
 * @param desiredOutput The desired output vector of the neural network.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
double trainOnData(struct NeuralNetwork nn, double * input, double * desiredOutput, double lrw, double lrb) {
    
    inputDataToNeuralNetwork(nn, input);
    
    double cost = costFunction(nn.outputVector, desiredOutput, nn.nrOfOutputNeurons);
    
    backPropogate(nn, desiredOutput, lrw, lrb);

    return cost;
}
