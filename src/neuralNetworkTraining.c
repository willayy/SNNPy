#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

void freeMatrix(double ** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

double * computeBiasGradients(struct NeuralNetwork * nn, double * partialGradients) {

    double * gradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons- nn->nrOfOutputNeurons);

    // the derivatives of the cost function with respect to the biases first to second to last layer.
    vectorReplace(gradients, partialGradients, nn->nrOfNeurons - nn->nrOfOutputNeurons);

    return gradients;
}

double ** computeWeightGradients(struct NeuralNetwork * nn, double * partialGradients) {

    double ** gradients = (double **) malloc(sizeof(double *) * nn->nrOfNeurons);

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        if (i < nn->nrOfNeurons - nn->neuronsPerLayer - nn->nrOfOutputNeurons ) {
            gradients[i] = (double *) malloc(sizeof(double) * nn->neuronsPerLayer);
        } else {
            gradients[i] = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);
        }
    }

    // the derivatives of the cost function with respect to the weights first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * neuronActivations = findConnectedNeuronActivations(nn, i);
        int * partialGradientIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdW = neuronActivations[j];
            gradients[i][j] = (partialGradients[partialGradientIndexes[j]] * dZdW);
        }
    }

    return gradients;

}

double * computePartialGradients(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA costFunctionDerivative) {

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);
    
    // Computations are made withing the neural network struct, with the values are already set from the forward propogation / input .

    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons; i--) {
        double neuronA = nn->neuronActivationVector[i];
        double neuronZ = nn->neuronValueVector[i];
        double dCdA = costFunctionDerivative(neuronA, desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ);
        partialGradients[i] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons; i >= 0; i--) {
        double neuronA = nn->neuronActivationVector[i];
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

void nudgeWeight(double * weight, double gradient, double lrw) {

    weight[0] -= lrw * gradient; 
}

void nudgeBias(double * bias, double gradient, double lrb) {

    bias[0] -= lrb * gradient;
}

void optimize(struct NeuralNetwork * nn, double ** Wgrad, double * Bgrad, double lrw, double lrb) {

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * weights = findOutputWeights(nn, i);
        for (int j = 0; j < nn->neuronsPerLayer; j++) {
            nudgeWeight(&weights[j], Wgrad[i][j], lrw);
        }
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * bias = &nn->biasVector[i];
        nudgeBias(bias, Bgrad[i], lrb);
    }
}

double ** averageWeightGradients(struct NeuralNetwork * nn, double *** sumGradients, double batchSize) {

    double ** averageGradients = (double **) malloc(sizeof(double *) * batchSize);

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        if (i < nn->nrOfNeurons - nn->neuronsPerLayer - nn->nrOfOutputNeurons ) {
            averageGradients[i] = (double *) malloc(sizeof(double) * nn->neuronsPerLayer);
            for (int j = 0; j < nrOfConnectedNeurons; j++) {
                averageGradients[i][j] = 0;
            }
        } else {
            averageGradients[i] = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);
            for (int j = 0; j < nrOfConnectedNeurons; j++) {
                averageGradients[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < batchSize; j++) {
            vectorAdd(averageGradients[i], sumGradients[j][i], nrOfConnectedNeurons);
        }
        vectorDiv(averageGradients[i], batchSize, nrOfConnectedNeurons);
    }

    return averageGradients;
}

double * averageBiasGradients(struct NeuralNetwork * nn, double ** sumGradients, double batchSize) {

    double * averageGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        averageGradients[i] = 0;
    }

    for (int i = 0; i < batchSize; i++) {
        vectorAdd(averageGradients, sumGradients[i], nn->nrOfNeurons);
    }

    vectorDiv(averageGradients, batchSize, nn->nrOfNeurons);

    return averageGradients;
}