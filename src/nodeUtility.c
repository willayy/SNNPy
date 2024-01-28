#include "neuralNetworkStructs.h"
#include "nodeUtility.h"
#include <stdlib.h>

int getNeuronLayer(struct NeuralNetwork nn, int neuron) {

    int isParamNeuron = (neuron < nn.nrOfParameters);
    int isHiddenNeuron =  (( neuron >= nn.nrOfParameters) && (neuron < (nn.nrOfParameters + nn.neuronsPerLayer*(nn.nrOfHiddenLayers-1))));
    int isHiddenToOutputNeuron = (neuron >= (nn.nrOfParameters + nn.nrOfHiddenNeurons - nn.neuronsPerLayer));

    if (isParamNeuron) {
        return 1;
    }

    else if (isHiddenNeuron) {
        return 2;
    }

    else if (isHiddenToOutputNeuron) {
        return 3;
    }
}

/**
 * Returns the number of nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the number of connected nodes of (node 0 is the top parameter node).
 * @return The number of connected nodes. */
int numberOfConnectedNeurons(struct NeuralNetwork nn, int neuron) {

    int nodeLayer = getNeuronLayer(nn, neuron);

    if (nodeLayer == 1) {
        return nn.neuronsPerLayer;
    }

    else if (nodeLayer == 2) {
        return nn.neuronsPerLayer;
    }

    else if (nodeLayer == 3) {
        return nn.nrOfOutputs;
    }
}

/**
 * Returns a vector of the activation values of the hidden neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getHiddenNeuronActivationValues(struct NeuralNetwork nn) {

    double * neuronActivationValues = (double *) malloc(sizeof(double)*nn.nrOfHiddenNeurons);

    for (int i = 0; i < nn.nrOfHiddenNeurons; ++i) {
        neuronActivationValues[i] = nn.neuronVector[i];
    }

    return neuronActivationValues;
}

/**
 * Returns a vector of the activation values of the parameter neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getParameterNeuronActivationValues(struct NeuralNetwork nn) {

    double * neuronActivationValues = (double *) malloc(sizeof(double)*nn.nrOfParameters);

    for (int i = 0; i < nn.nrOfParameters; ++i) {
        neuronActivationValues[i] = nn.parameterVector[i];
    }

    return neuronActivationValues;
}

/**
 * Returns a vector of the activation values of the output neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getOutputNeuronActivationValues(struct NeuralNetwork nn) {

    double * neuronActivationValues = (double *) malloc(sizeof(double)*nn.nrOfOutputs);

    for (int i = 0; i < nn.nrOfOutputs; ++i) {
        neuronActivationValues[i] = nn.outputVector[i];
    }

    return neuronActivationValues;
}

/**
 * Returns a vector of the activation values of all neurons in the neural network.
 * @param nn The neural network.
 * @return A vector of the activation values of the neurons in the neural network. */
double * getNeuronActiviationValues(struct NeuralNetwork nn) {

    double * neuronActivationValues = (double *) malloc(sizeof(double)*(nn.nrOfParameters + nn.nrOfHiddenNeurons + nn.nrOfOutputs));

    double * parameterNeuronActivationValues = getParameterNeuronActivationValues(nn);
    double * hiddenNeuronActivationValues = getHiddenNeuronActivationValues(nn);
    double * outputNeuronActivationValues = getOutputNeuronActivationValues(nn);

    for (int i = 0; i < nn.nrOfParameters; ++i) {
        neuronActivationValues[i] = parameterNeuronActivationValues[i];
    }

    for (int i = 0; i < nn.nrOfHiddenNeurons; ++i) {
        neuronActivationValues[i + nn.nrOfParameters] = hiddenNeuronActivationValues[i];
    }

    for (int i = 0; i < nn.nrOfOutputs; ++i) {
        neuronActivationValues[i + nn.nrOfParameters + nn.nrOfHiddenNeurons] = outputNeuronActivationValues[i];
    }

    free(parameterNeuronActivationValues);
    free(hiddenNeuronActivationValues);
    free(outputNeuronActivationValues);

    return neuronActivationValues;
}

/**
 * Returns pointer to a neuron in the neural network, cant be a output neuron.
 * @param nn The neural network.
 * @param node The neuron. */
double * findNeuron(struct NeuralNetwork nn, int neuron) {

    int nodeLayer = getNeuronLayer(nn, neuron);
    
    if (nodeLayer == 1) {
        return & nn.parameterVector[neuron];
    }

    else if (nodeLayer == 2) {
        neuron = neuron - nn.nrOfParameters;
        return & nn.neuronVector[neuron];
    }

    else if (nodeLayer == 3) {
        neuron = neuron - nn.nrOfParameters;
        return & nn.neuronVector[neuron];
    }
}

/**
 * returns a vector of the nodes connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected nodes of (node 0 is the top parameter node).
 * @return A vector of the connected nodes. */
double ** findConnectedNeurons(struct NeuralNetwork nn, int neuron) {

    double ** connectedNodes = (double **) malloc(sizeof(double*)*nn.neuronsPerLayer);

    int nodeLayer = getNeuronLayer(nn, neuron);

    if (nodeLayer == 1) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            connectedNodes[i] = & nn.neuronVector[i];
        }
    }

    else if (nodeLayer == 2) {
        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            int offset = neuron - nn.nrOfParameters;
            connectedNodes[i] = & nn.neuronVector[offset + i];
        }
    }

    else if (nodeLayer == 3) {
        for (int i = 0; i < nn.nrOfOutputs; ++i) {
            connectedNodes[i] = & nn.outputVector[i];
        }
    }

    return connectedNodes;
}

/**
 * returns a vector of the weights connected to a given node in the forward direction.
 * @param nn The neural network.
 * @param node The node to find the connected weights of (node 0 is the top parameter node).
 * @return A vector of the connected weights. */
double ** findConnectedWeights(struct NeuralNetwork nn, int neuron) {
    
    double ** connectedWeights = (double **) malloc(sizeof(double *)*(nn.neuronsPerLayer*nn.neuronsPerLayer));

    int nodeLayer = getNeuronLayer(nn, neuron);

    if (nodeLayer == 1) {

        int offset = neuron * nn.neuronsPerLayer;

        for (int i = 0; i < nn.neuronsPerLayer; ++i) {

            connectedWeights[i] = & nn.weightMatrix[offset + i];
        }
    }

    else if (nodeLayer == 2) {

        int nodeLayer = (neuron - nn.nrOfParameters) / nn.neuronsPerLayer;

        int parameterLayerWeights = nn.nrOfParameters * nn.neuronsPerLayer;

        int offset = nodeLayer * nn.weightsPerLayer + parameterLayerWeights + (neuron - nn.nrOfParameters) * nn.neuronsPerLayer;

        for (int i = 0; i < nn.neuronsPerLayer; ++i) {
            connectedWeights[i] = & nn.weightMatrix[offset + i];
        }
    }

    else if (nodeLayer == 3) {

        int parameterLayerWeights = nn.nrOfParameters * nn.neuronsPerLayer;

        int hiddenLayerWeights = nn.weightsPerLayer * (nn.nrOfHiddenLayers - 1);

        int offset = parameterLayerWeights + hiddenLayerWeights + (neuron - nn.nrOfParameters - nn.neuronsPerLayer * (nn.nrOfHiddenLayers - 1)) * nn.nrOfOutputs;
        for (int i = 0; i < nn.nrOfOutputs; ++i) {
            connectedWeights[i] = & nn.weightMatrix[offset + i];
        }
    }

    return connectedWeights;
}