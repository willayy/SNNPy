#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"



double * computeGradientsBiases(struct NeuralNetwork * nn, const double * partialGradients, double batchSize) {

    vectorReplace(nn->neuronActivationVector, partialGradients, nn->nrOfNeurons);

    double * gradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);

    // the derivatives of the cost function with respect to the biases first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double const * partialGradient = findConnectedNeuronActivations(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            gradients[i] = partialGradient[j] / batchSize; // dZdB = 1
        }
    }

    return gradients;
}

double ** computeGradientsWeights(struct NeuralNetwork * nn, const double * partialGradients, double batchSize) {

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
            gradients[i][j] = (partialGradient[j] * dZdW) / batchSize;
        }
    }

    free(neuronActivations);

    return gradients;

}

double * computePartialGradient(struct NeuralNetwork * nn, const double * output, const double * desiredOutput, dblA_dblA costFunctionDerivative) {

    vectorReplace(nn->neuronActivationVector, output, nn->nrOfOutputNeurons);

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);
    
    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons; i < nn->nrOfNeurons; i++) {
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double dCdA = costFunctionDerivative(neuronA[0], desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ[0]);
        neuronA[0] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - nn->neuronsPerLayer; i > 0; i--) {
        double dZdA;
        double dAdZ;
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double * connectedneurons = findConnectedNeuronActivations(nn, i);
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double gradientSum = 0;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            dZdA = connectedneurons[j] * weights[j];
            dAdZ = nn->activationFunctionDerivative(neuronZ[0]);
            gradientSum += dZdA * dAdZ;
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

void optimize(struct NeuralNetwork * nn, const double ** Wgrad, const double * Bgrad, const double lrw, const double lrb) {

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

