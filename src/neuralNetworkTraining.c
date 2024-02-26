#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

void freeWeightGradients(double ** gradients, int nrOfNeurons) {

    for (int i = 0; i < nrOfNeurons; i++) {
        free(gradients[i]);
    }

    free(gradients);
}

double * computeBiasGradients(struct NeuralNetwork * nn, double * partialGradients) {

    vectorReplace(nn->neuronActivationVector, partialGradients, nn->nrOfNeurons);

    double * gradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);

    // the derivatives of the cost function with respect to the biases first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double const * partialGradient = findConnectedNeuronActivations(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            gradients[i] = partialGradient[j]; // dZdB = 1
        }
    }

    return gradients;
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

double ** computeWeightGradients(struct NeuralNetwork * nn, double * partialGradients) {

    double * neuronActivations = (double *) malloc(sizeof(double) * nn->nrOfNeurons);
    
    vectorReplace(neuronActivations, nn->neuronActivationVector, nn->nrOfNeurons);

    vectorReplace(nn->neuronActivationVector, partialGradients, nn->nrOfNeurons);

    double ** gradients = (double **) malloc(sizeof(double *) * nn->nrOfNeurons);

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        if (i < nn->nrOfNeurons - nn->neuronsPerLayer - nn->nrOfOutputNeurons ) {
            gradients[i] = (double *) malloc(sizeof(double) * nn->neuronsPerLayer);
        } else {
            gradients[i] = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);
        }
    }

    // the derivatives of the cost function with respect to the weights first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double const * partialGradient = findConnectedNeuronActivations(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdW = neuronActivations[0];
            gradients[i][j] = (partialGradient[j] * dZdW);
        }
    }

    free(neuronActivations);

    return gradients;

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

double * computePartialGradient(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA costFunctionDerivative) {

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);
    
    // Computations are made withing the neural network struct, with the values are already set from the forward propogation / input .

    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons; i--) {
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double dCdA = costFunctionDerivative(neuronA[0], desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ[0]);
        neuronA[0] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons; i >= 0; i--) {
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double * weights = findOutputWeights(nn, i);
        double * connectedNeurons = findConnectedNeuronActivations(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double gradientSum = 0;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdA = weights[0];
            double dAdZ = nn->activationFunctionDerivative(neuronZ[0]);
            gradientSum += connectedNeurons[j] * dZdA * dAdZ;
        }
        neuronA[0] = gradientSum;
    }

    vectorReplace(partialGradients, nn->neuronActivationVector, nn->nrOfNeurons);

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
        double * biases = findBias(nn, i);
        nudgeBias(biases, Bgrad[i], lrb);
    }
}

