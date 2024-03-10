#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

void initNeuralNetworkGradients(NeuronGradient ** gradients, NeuralNetwork * nn) {
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfParameterNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        gradients[i] = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        gradients[i]->weightGradient = (double *) malloc(sizeof(double) * nrOfConnectedNeurons);
        gradients[i]->biasGradient = (double *) malloc(sizeof(double));
        vectorSet(gradients[i]->weightGradient, 0, nrOfConnectedNeurons);
        gradients[i]->biasGradient[0] = 0;
    }
}

NeuronGradient ** computeGradients(NeuralNetwork * nn, double * partialGradients) {

    NeuronGradient ** gradients = (NeuronGradient **) malloc(sizeof(NeuronGradient *) * (nn->nrOfNeurons - nn->nrOfParameterNeurons));

    initNeuralNetworkGradients(gradients, nn);

    // the derivatives of the cost function with respect to the weights and biases from first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        int * partialGradientIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double dZdW = nn->neuronActivationVector[i];
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            gradients[i]->weightGradient[j] = (partialGradients[partialGradientIndexes[j]] * dZdW);
            gradients[i]->biasGradient[0] = partialGradients[i];
        }
    }

    return gradients;
}

double * computePartialGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative) {

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);

    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i--) {
        double neuronA = nn->neuronActivationVector[i];
        double neuronZ = nn->neuronValueVector[i];
        double dCdA = costFunctionDerivative(neuronA, desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ);
        partialGradients[i] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i >= 0; i--) {
        double neuronZ = nn->neuronValueVector[i];
        double * weights = findOutputWeights(nn, i);
        int * connectedNeuronIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double gradientSum = 0;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdA = weights[j];
            double dAdZ = nn->activationFunctionDerivative(neuronZ);
            gradientSum += partialGradients[connectedNeuronIndexes[i]] * dZdA * dAdZ;
        }
        partialGradients[i] = gradientSum;
    }

    return partialGradients;
}

NeuronGradient ** averageGradients(NeuralNetwork * nn, NeuronGradient *** sumNg, int batchSize) {

    NeuronGradient ** avgNg = (NeuronGradient **) malloc(sizeof(NeuronGradient *) * nn->nrOfNeurons);

    initNeuralNetworkGradients(avgNg, nn);

    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < nn->nrOfNeurons - nn->nrOfOutputNeurons; j++) {
            int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, j);
            for (int k = 0; k < nrOfConnectedNeurons; k++) {
                avgNg[j]->weightGradient[k] += (sumNg[i][j]->weightGradient[k])*(1.0/batchSize);
            }
            avgNg[j]->biasGradient[0] += sumNg[i][j]->biasGradient[0] * (1.0/batchSize);
        }
    }

    return avgNg;
}

void optimize(NeuralNetwork * nn, NeuronGradient ** avgNg, double lrw, double lrb) {

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double * weight = &weights[j];
            double gradient = avgNg[i]->weightGradient[j];
            weight[0] -= lrw * gradient;
        }
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * bias = &nn->biasVector[i];
        double gradient = avgNg[i]->biasGradient[0];
        bias[0] -= lrb * gradient;
    }
}